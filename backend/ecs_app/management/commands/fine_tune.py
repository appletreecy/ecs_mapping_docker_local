from django.core.management.base import BaseCommand
from ecs_app.fine_tune import fine_tune_model

class Command(BaseCommand):
    help = "Fine-tune the SentenceTransformer model with confirmed ECS mappings"

    def handle(self, *args, **kwargs):
        self.stdout.write("🚀 Starting fine-tuning process...")
        fine_tune_model()
        self.stdout.write("✅ Fine-tuning completed successfully!")
