# core/views.py
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from datasets.models import Dataset
from agents.agent2 import analyze_csv_dataset, analyze_image_dataset
from datasets.views import ANALYSIS_RESULTS  # dictionnaire global pour l'analyse
import pandas as pd
import logging

import json

# ------------------------
# Page d'accueil publique
# ------------------------
def home_view(request):
    """
    Vue pour la page d'accueil introductive de Smart BI
    Cette page présente l'application et oriente l'utilisateur
    vers les fonctionnalités principales (Datasets, Agents, Analytics, Reports)
    """
    return render(request, 'core/home.html')


# ------------------------
# Page d'accueil utilisateur connecté
# ------------------------
@login_required
def home_user_view(request):
    """
    Affiche la liste des datasets uploadés par l'utilisateur
    """
    datasets = Dataset.objects.filter(user=request.user).order_by('-uploaded_at')
    context = {
        "user": request.user,
        "datasets": datasets
    }
    return render(request, "core/home_user.html", context)


# ------------------------
# Vérification AJAX de l'analyse
# ------------------------
logger = logging.getLogger(__name__)

@login_required
def check_analysis_status(request, dataset_id):
    """
    AJAX endpoint pour retourner le status de l'analyse.
    Toujours retourne du JSON pour éviter les erreurs JS.
    """
    try:
        result = ANALYSIS_RESULTS.get(dataset_id)

        if result:
            response_data = {"status": "done", "result": result}
            # Supprimer pour éviter saturation mémoire
            del ANALYSIS_RESULTS[dataset_id]
        else:
            response_data = {"status": "pending"}

    except Exception as e:
        response_data = {"status": "error", "message": str(e)}

    # 🔹 Debug: afficher le JSON dans la console du serveur
    print(f"[DEBUG] check_analysis_status JSON for dataset_id={dataset_id}: {response_data}")

    return JsonResponse(response_data)



# ------------------------
# Dashboard BI
# ------------------------
@login_required
@login_required
def dashboard_view(request, dataset_id):
    dataset = Dataset.objects.get(id=dataset_id)
    dataset_path = dataset.file.path
    filename = dataset_path.lower()

    try:
        if filename.endswith('.csv'):
            agent_result = analyze_csv_dataset(dataset_path)
            dataset_type = "csv"
            preview = pd.read_csv(dataset_path).head(5).to_dict(orient="records")
        elif filename.endswith('.zip'):
            agent_result = analyze_image_dataset(dataset_path)
            dataset_type = "image"
            preview = []
        else:
            agent_result = {"status": "error", "message": "Seuls CSV ou ZIP d'images sont supportés"}
            dataset_type = "error"
            preview = []
    except Exception as e:
        agent_result = {"status": "error", "message": str(e)}
        dataset_type = "error"
        preview = []

    # Assurez-vous que tout est JSON-friendly
    def make_json_safe(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_list()
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_json_safe(v) for v in obj]
        return str(obj)  # fallback

    agent_result_safe = make_json_safe(agent_result)

    return render(request, "core/dashboard_bi.html", {
        "agent2_result_json": json.dumps(agent_result_safe),  # nom correct pour le template
        "dataset_name": dataset.name,
        "dataset_type": dataset_type,
        "preview": preview,
    })



# ------------------------
# Page de loading avec design
# ------------------------
@login_required
def loading_view(request, dataset_id):
    """
    Affiche la page de loading avec design moderne.
    Vérifie périodiquement si l'analyse est terminée via AJAX.
    """
    # Construire l'URL de vérification AJAX
    check_status_url = f"/check_status/{dataset_id}/"
    return render(request, "core/loading.html", {"loading_url": check_status_url})
