import faiss
import pickle
import numpy as np
from django.http import JsonResponse
from rest_framework.decorators import api_view
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from .models import ECSMapping, ECSMappingFeedback
from .config import OPENAI_API_KEY
import logging
from rest_framework.response import Response

logger = logging.getLogger("ecs_app")

# Load AI Models
logger.info("🔄 Loading AI Models...")
model = SentenceTransformer("all-MiniLM-L6-v2")
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
logger.info("✅ AI Models Loaded Successfully!")


# Fetch all stored ECS mappings
def fetch_all_mappings():
    logger.debug("📥 Fetching all ECS mappings from the database...")
    mappings = ECSMapping.objects.all().values("log_field", "ecs_field", "embedding", "description",
                                               "example_log_values")
    logger.debug(f"📋 Retrieved {len(mappings)} mappings from the database.")
    return list(mappings)


# Initialize FAISS
def initialize_faiss():
    logger.info("🔄 Initializing FAISS index...")
    mappings = fetch_all_mappings()

    if not mappings:
        logger.warning("⚠️ No mappings found! FAISS index cannot be initialized.")
        return None

    embeddings = np.vstack([pickle.loads(m["embedding"]) for m in mappings]).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    logger.info(f"✅ FAISS index initialized with {len(mappings)} entries.")
    return index


# Retrieve FAISS index
def get_faiss_index():
    logger.debug("📡 Retrieving FAISS index...")
    return initialize_faiss()


# Hybrid Search: FAISS (Embeddings) + BM25 (Text Matching)
def find_similar_fields_hybrid(log_field, top_k=3):
    """Hybrid search using FAISS + BM25."""
    logger.info(f"🔍 Searching for similar ECS mappings for: {log_field}")

    faiss_index = get_faiss_index()
    if faiss_index is None:
        logger.warning("⚠️ FAISS index is empty. No similar fields found.")
        return []

    query_embedding = model.encode([log_field]).astype("float32")
    distances, indices = faiss_index.search(query_embedding, top_k)

    mappings = fetch_all_mappings()
    corpus = [m["log_field"] for m in mappings]
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(word_tokenize(log_field.lower()))

    matched_fields = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(mappings):
            similarity_score = 1 / (1 + dist)
            bm25_score = bm25_scores[idx] / max(bm25_scores) if max(bm25_scores) > 0 else 0
            final_score = (0.7 * similarity_score) + (0.3 * bm25_score)

            matched_fields.append((mappings[idx]["log_field"], mappings[idx]["ecs_field"], final_score))

    matched_fields.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"✅ Found {len(matched_fields)} similar mappings for '{log_field}'.")

    return matched_fields[:top_k]


def chatgpt_ecs_mapping(log_field, similar_fields):
    """Use ChatGPT to determine ECS mapping with few-shot learning."""
    logger.info(f"🤖 Calling ChatGPT for ECS mapping of: {log_field}")

    few_shot_example = """
    Example Mappings:
    - source_ip -> source.address
    - destination_ip -> destination.address
    - user_agent -> user_agent.original
    """

    similar_text = "\n".join([f"{sf[0]} -> {sf[1]}" for sf in similar_fields])

    messages = [
        SystemMessage(content="You are an expert in log processing and ECS mapping."),
        HumanMessage(content=f"""
        {few_shot_example}

        Here are existing mappings:
        {similar_text}

        Now, map the following log field to its ECS equivalent:
        '{log_field}'

        Provide ONLY the ECS field name or return 'none_ecs_field' if no exact match exists.

        """)
    ]

    response = llm(messages)
    ecs_field = response.content.strip()

    logger.info(f"📢 ChatGPT Response: {ecs_field}")
    return ecs_field if ecs_field.lower() != "none" else "none_ecs_field"


# Store confidence scores and insert into database
def insert_mapping(log_field, ecs_field, embedding, confidence_score=0.8):
    """Insert or update ECS mapping with confidence score."""
    logger.info(f"💾 Storing ECS mapping: {log_field} -> {ecs_field} (Confidence: {confidence_score:.2f})")

    embedding_binary = pickle.dumps(embedding)
    mapping, created = ECSMapping.objects.update_or_create(
        log_field=log_field,
        defaults={"ecs_field": ecs_field, "embedding": embedding_binary, "confidence_score": confidence_score}
    )

    logger.info(f"{'✅ Created' if created else '🔄 Updated'} Mapping: {log_field} -> {ecs_field}")

# Retrieve stored mapping from Database
def get_stored_mapping(log_field):
    """Retrieve stored ECS mapping from Django ORM."""
    mapping = ECSMapping.objects.filter(log_field=log_field).first()
    return mapping.ecs_field if mapping else None

# API Endpoint: Get or Create ECS Mapping
@api_view(['POST'])
def get_ecs_mapping(request):
    """API Endpoint to retrieve or create an ECS mapping for given log fields."""
    logger.info("📩 Received API request for ECS mapping.")

    data = request.data.get("log_field", [])
    response_dict = {}

    for log_field in data:
        logger.debug(f"🔍 Processing log field: {log_field}")

        stored_mapping = get_stored_mapping(log_field)
        if stored_mapping:
            logger.info(f"✅ Using stored mapping: {log_field} -> {stored_mapping}")
            response_dict[log_field] = stored_mapping
            continue

        similar_fields = find_similar_fields_hybrid(log_field)
        if similar_fields and similar_fields[0][2] >= 0.75:
            response_dict[log_field] = similar_fields[0][1]  # Use top FAISS+BM25 match
            logger.info(f"🎯 Found similar mapping: {log_field} -> {similar_fields[0][1]}")
        else:
            new_mapping = chatgpt_ecs_mapping(log_field, similar_fields)
            new_embedding = model.encode([log_field]).astype("float32")
            insert_mapping(log_field, new_mapping, new_embedding)
            response_dict[log_field] = new_mapping
            logger.info(f"📢 ChatGPT-generated mapping stored: {log_field} -> {new_mapping}")

    return Response(response_dict)


# API Endpoint: Store User Feedback on ECS Mapping
@api_view(['POST'])
def submit_feedback(request):
    """API Endpoint to collect user feedback on ECS mapping."""
    logger.info("📩 Received user feedback.")

    data = request.data
    log_field = data.get("log_field")
    ecs_field = data.get("ecs_field")
    correct = data.get("correct")

    if not log_field or not ecs_field or correct is None:
        logger.warning("⚠️ Missing required fields in feedback submission.")
        return Response({"error": "Missing required fields"}, status=400)

    ECSMappingFeedback.objects.create(log_field=log_field, ecs_field=ecs_field, correct=correct)
    logger.info(f"✅ Feedback stored: {log_field} -> {ecs_field} (Correct: {correct})")

    return Response({"message": "Feedback submitted successfully"})
