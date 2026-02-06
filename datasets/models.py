from django.db import models
from acountes.models import CustomUser

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to="datasets/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='datasets')
    analysis_json = models.JSONField(null=True, blank=True)



