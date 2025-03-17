import os
import logging
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from django.conf import settings
from django.db.models import Q
from .models import ECSMapping

# Fix multiprocessing issue on macOS (Apple MPS Backend)
try:
    torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Avoids re-initialization error in forked processes

logger = logging.getLogger("ecs_app")

def fine_tune_model():
    """Fine-tune SentenceTransformer using confirmed ECS mappings."""

    logger.info("🔄 Starting fine-tuning process for SentenceTransformer...")

    # Step 1: Fetch confirmed ECS mappings (confidence_score >= 0.7)
    logger.info("📥 Fetching confirmed ECS mappings from the database...")
    dataset = ECSMapping.objects.filter(Q(confidence_score__gte=0.7)).values_list("log_field", "ecs_field")

    if not dataset:
        logger.warning("⚠️ No confirmed mappings found! Skipping fine-tuning.")
        return

    training_data = [InputExample(texts=[log, ecs]) for log, ecs in dataset]
    logger.info(f"✅ Loaded {len(training_data)} training examples.")

    # Step 2: Load Pretrained Model
    logger.info("📡 Loading base SentenceTransformer model: 'all-MiniLM-L6-v2'...")

    # Enforce CPU usage if running on macOS MPS (fix for stability)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    logger.info(f"✅ Model loaded successfully. Running on: {device}")

    # Step 3: Create DataLoader
    logger.info("📦 Preparing DataLoader for training...")
    train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)
    logger.info("✅ DataLoader is ready.")

    # Step 4: Define Loss Function
    logger.info("⚙️ Defining loss function: MultipleNegativesRankingLoss...")
    train_loss = losses.MultipleNegativesRankingLoss(model)
    logger.info("✅ Loss function initialized.")

    # Step 5: Start Fine-Tuning
    logger.info("🚀 Fine-tuning model for 1 epoch with warmup steps...")

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    logger.info("✅ Fine-tuning completed.")

    # Step 6: Save Fine-Tuned Model
    save_path = os.path.join(settings.BASE_DIR, "fine_tuned_model")
    logger.info(f"💾 Saving fine-tuned model to '{save_path}'...")
    model.save(save_path)
    logger.info("✅ Model saved successfully.")

    logger.info("🎉 Fine-tuning process completed successfully!")
