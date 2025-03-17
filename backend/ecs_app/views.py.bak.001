import faiss
import pickle
import numpy as np
from django.http import JsonResponse
from rest_framework.decorators import api_view
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from .models import ECSMapping  # ✅ Using Django ORM

from .config import  OPENAI_API_KEY

# Load AI Models
model = SentenceTransformer("all-MiniLM-L6-v2")
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)


# Fetch all stored ECS mappings using Django ORM
def fetch_all_mappings():
    """Fetch all ECS mappings from the database using Django ORM."""
    mappings = ECSMapping.objects.all().values("log_field", "ecs_field", "embedding")
    return list(mappings)  # Convert QuerySet to list


# Insert or update ECS mapping using Django ORM
def insert_mapping(log_field, ecs_field, embedding):
    """Insert or update ECS mapping using Django ORM."""
    embedding_binary = pickle.dumps(embedding)  # Serialize embedding
    mapping, created = ECSMapping.objects.update_or_create(
        log_field=log_field,
        defaults={"ecs_field": ecs_field, "embedding": embedding_binary},
    )

    if created:
        print(f"✅ New ECS mapping created: {log_field} -> {ecs_field}")
    else:
        print(f"🔄 ECS mapping updated: {log_field} -> {ecs_field}")


# Always initialize FAISS from MySQL dynamically using ORM
def initialize_faiss():
    mappings = fetch_all_mappings()
    if not mappings:
        print("⚠️ FAISS Initialization: No mappings found in database.")
        return None  # No existing mappings yet

    embeddings = np.vstack([pickle.loads(m["embedding"]) for m in mappings]).astype("float32")

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"✅ FAISS Rebuilt with {len(mappings)} mappings!")
    return index


# Load FAISS (Always from ORM)
def get_faiss_index():
    return initialize_faiss()


# Find similar ECS mappings using FAISS (with similarity threshold)
def find_similar_fields(log_field, top_k=3, similarity_threshold=0.8):
    """Find similar ECS mappings using FAISS similarity search."""
    faiss_index = get_faiss_index()
    if faiss_index is None:
        print("⚠️ No FAISS index found. Returning no matches.")
        return []  # No reference data yet

    query_embedding = model.encode([log_field]).astype("float32")
    distances, indices = faiss_index.search(query_embedding, top_k)

    mappings = fetch_all_mappings()
    matched_fields = []

    print(f"🔎 Searching FAISS for: {log_field}")

    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(mappings):  # Ensure index is valid
            similarity_score = 1 / (1 + dist)  # Convert distance to similarity score
            matched_log_field = mappings[idx]["log_field"]
            matched_ecs_field = mappings[idx]["ecs_field"]

            # ✅ Apply similarity threshold
            if similarity_score >= similarity_threshold:
                print(f"✅ FAISS Match Found (High Similarity): {matched_log_field} -> {matched_ecs_field} (Score: {similarity_score:.2f})")
                matched_fields.append((matched_log_field, matched_ecs_field))
            else:
                print(f"❌ FAISS Match Rejected (Low Similarity): {matched_log_field} -> {matched_ecs_field} (Score: {similarity_score:.2f})")

    return matched_fields

# Find similar ECS mappings using FAISS (with similarity threshold)
def find_similar_fields_for_ecs_mapping(log_field, top_k=3, similarity_threshold=0.8):
    """Find similar ECS mappings using FAISS similarity search."""
    faiss_index = get_faiss_index()
    if faiss_index is None:
        print("⚠️ No FAISS index found. Returning no matches.")
        return []  # No reference data yet

    query_embedding = model.encode([log_field]).astype("float32")
    distances, indices = faiss_index.search(query_embedding, top_k)

    mappings = fetch_all_mappings()
    matched_fields = []

    print(f"🔎 Searching FAISS for: {log_field}")

    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(mappings):  # Ensure index is valid
            similarity_score = 1 / (1 + dist)  # Convert distance to similarity score
            matched_log_field = mappings[idx]["log_field"]
            matched_ecs_field = mappings[idx]["ecs_field"]

            # ✅ Apply similarity threshold
            if similarity_score >= similarity_threshold:
                print(f"✅ FAISS Match Found (High Similarity): {matched_log_field} -> {matched_ecs_field} (Score: {similarity_score:.2f})")
                matched_fields.append((matched_log_field, matched_ecs_field))
            else:
                print(f"❌ FAISS Match Rejected (Low Similarity): {matched_log_field} -> {matched_ecs_field} (Score: {similarity_score:.2f})")
                matched_fields.append((matched_log_field, matched_ecs_field))

    return matched_fields


# Use LangChain ChatGPT for ECS mapping
def chatgpt_ecs_mapping(log_field, similar_fields):
    """Use ChatGPT to determine the correct ECS mapping for an unknown log field, ensuring only valid ECS fields are returned."""
    similar_text = "\n".join([f"{sf[0]} -> {sf[1]}" for sf in similar_fields])

    messages = [
        SystemMessage(content="You are an expert in mapping log fields to Elastic Common Schema (ECS). Your job is to provide the correct ECS field name."),
        HumanMessage(content=f"Here are existing mappings:\n{similar_text}\n\nNow, map the following log field to its ECS equivalent:\n'{log_field}'\n\nProvide ONLY the ECS field name. If no direct match is available, return exactly 'none_ecs_field'. Do not provide explanations or additional text.")
    ]

    response = llm(messages)
    ecs_field = response.content.strip()

    print(f"ecs_field is {ecs_field}")

    # Ensure the response is a valid ECS field or "none_ecs_field"
    if ecs_field.lower() in ["none", "none_ecs_field", "no exact match", "not available", "note that ECS does not have a specific field for"]:
        ecs_field = "none_ecs_field"

    print(f"ChatGPT Mapping Response: {ecs_field}")  # ✅ Debugging log

    return ecs_field



# Get stored ECS mapping using ORM
def get_stored_mapping(log_field):
    """Retrieve ECS mapping using Django ORM."""
    mapping = ECSMapping.objects.filter(log_field=log_field).first()
    return mapping.ecs_field if mapping else None


# API: Get or Create ECS Mapping
@api_view(['POST'])
def get_ecs_mapping(request):
    """API Endpoint to retrieve or create an ECS mapping for a given log field."""
    datas = request.data

    print(datas)

    response_dict = {}

    for data in datas["log_field"]:
        log_field = data.strip()

        if not log_field:
            return JsonResponse({"error": "log_field is required"}, status=400)

        stored_mapping = get_stored_mapping(log_field)
        if stored_mapping:
            print(f"Using stored mapping: {log_field} -> {stored_mapping}")
            # return JsonResponse({"log_field": log_field, "ecs_field": stored_mapping})

            response_dict[log_field] = stored_mapping

        # Find similar fields using FAISS
        similar_fields = find_similar_fields(log_field, top_k=3)

        if similar_fields:
            print(f"📌 FAISS returned: {similar_fields}")  # Debugging output
        else:
            print("❌😵"
                  " FAISS found no similar fields.")

        # If no strong match is found, use ChatGPT
        if not similar_fields:

            similar_fields_for_ecs_mapping = find_similar_fields_for_ecs_mapping(log_field, top_k=3)

            new_mapping = chatgpt_ecs_mapping(log_field, similar_fields_for_ecs_mapping)

            # Generate embedding for the log field
            new_embedding = model.encode([log_field]).astype("float32")

            # print(f"New embedding is: {new_embedding}")

            print(f"new_mapping is {new_mapping}")

            new_mapping = new_mapping.replace("'", "")

            # Save to MySQL using Django ORM
            insert_mapping(log_field, new_mapping, new_embedding)

            print(f"New mapping stored: {log_field} -> {new_mapping}")

            response_dict[log_field] = new_mapping
            # return JsonResponse({"log_field": log_field, "ecs_field": new_mapping}) # ✅

        if similar_fields:

            response_dict[log_field] = similar_fields[0][1]

        # return JsonResponse({"log_field": log_field, "ecs_field": similar_fields[0][1]})  # Return the best FAISS match

    print(response_dict)

    return JsonResponse(response_dict)
