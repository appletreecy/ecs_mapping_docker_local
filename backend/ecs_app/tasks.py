from celery import shared_task
from .fine_tune import fine_tune_model
import logging

logger = logging.getLogger("ecs_app")

@shared_task
def train_model_task():
    logger.info("🚀 Starting scheduled fine-tuning...")
    fine_tune_model()
    logger.info("✅ Fine-tuning completed successfully!")