from django.shortcuts import render
from django.http import JsonResponse
from pathlib import Path
import pandas as pd

from datasets.models import Dataset
from agents.agent1 import analyze_columns
from agents.agent2 import analyze_sales_dataset

def agent2_analysis_view(request, dataset_id):
    """
    Lance l'analyse Agent2 pour un dataset 
    """
    try:
        dataset = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        return JsonResponse({"status": "error", "message": "Dataset introuvable."}, status=404)

    dataset_path = Path(dataset.file.path)


    try:
        result = analyze_csv_dataset(dataset_path,dataset_id)
    except Exception as e:
        return JsonResponse({"status": "error", "message": f"Agent2 failed: {str(e)}"}, status=500)

    return JsonResponse({
        "status": "success",
        "dataset_id": dataset.id,
        "file_name": dataset.name,
        "agent2_result": result
    })
