from django.db import models
from datasets.models import Dataset

class ChatRequest(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField(null=True, blank=True)
    status = models.CharField(
        max_length=120,
        default="pending"
    )  # pending | done | error
    created_at = models.DateTimeField(auto_now_add=True)
