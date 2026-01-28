from django import forms
from .models import Dataset

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['file','name']  # seul champ nécessaire
        widgets = {
            'file': forms.ClearableFileInput(attrs={
                'class': 'form-control',
                'accept': '.csv'
            })
        }
        labels = {
            'file': 'Upload your CSV Dataset'
        }
