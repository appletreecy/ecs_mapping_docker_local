from django.db import models

class ECSMapping(models.Model):
    log_field = models.CharField(max_length=255, unique=True)
    ecs_field = models.CharField(max_length=255)
    embedding = models.BinaryField() # Store FAISS embedding as binary
    description = models.TextField(null=True, blank=True)
    example_log_values = models.TextField(null=True, blank=True)
    confidence_score = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.log_field} -> {self.ecs_field}"

class ECSMappingFeedback(models.Model):
    log_field = models.CharField(max_length=255)
    ecs_field = models.CharField(max_length=255)
    correct = models.BooleanField(default=True) # True = Correct False = Incorrect



