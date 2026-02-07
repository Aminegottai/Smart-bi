from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from datasets.models import Dataset
from agents.agent2 import analyze_csv_dataset, analyze_image_dataset
from datasets.views import ANALYSIS_RESULTS
import pandas as pd
import json
import math
from django.views.decorators.csrf import csrf_exempt
import threading
from .chatbot import process_llm, llm_results_cache

# ------------------------
# Fonction de nettoyage NaN/Infinity
# ------------------------
def clean_nan_inf(obj):
    """
    Remplace récursivement NaN et Infinity par None (null en JSON).
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_nan_inf(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_inf(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.fillna(0).replace([float('inf'), float('-inf')], 0).to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.fillna(0).replace([float('inf'), float('-inf')], 0).tolist()
    return obj


# ------------------------
# Encodeur JSON personnalisé (backup)
# ------------------------
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return super().default(obj)


# ------------------------
# Page d'accueil publique
# ------------------------
def home_view(request):
    return render(request, 'core/home.html')


# ------------------------
# Page d'accueil utilisateur
# ------------------------
@login_required
def home_user_view(request):
    datasets = Dataset.objects.filter(user=request.user).order_by('-uploaded_at')
    return render(request, "core/home_user.html", {"datasets": datasets, "user": request.user})


# ------------------------
# Vérification AJAX de l'analyse
# ------------------------
@login_required
def check_analysis_status(request, dataset_id):
    try:
        result = ANALYSIS_RESULTS.get(dataset_id)
        if result:
            # ✅ Nettoie NaN/Infinity avant de retourner
            result = clean_nan_inf(result)
            response_data = {"status": "done", "result": result}
            del ANALYSIS_RESULTS[dataset_id]
        else:
            response_data = {"status": "pending"}
    except Exception as e:
        response_data = {"status": "error", "message": str(e)}
    
    return JsonResponse(response_data, safe=False)


# ------------------------
# Dashboard BI
# ------------------------
@login_required
def dashboard_view(request, dataset_id):
    dataset = Dataset.objects.get(id=dataset_id)
    dataset_path = dataset.file.path
    filename = dataset_path.lower()

    try:
        if filename.endswith(".csv"):
            agent_result = analyze_csv_dataset(dataset_path, dataset_id)
            dataset_type = "csv"
            preview = pd.read_csv(dataset_path).head(5)
            # ✅ Nettoie preview
            preview = clean_nan_inf(preview.to_dict(orient="records"))
        elif filename.endswith(".zip"):
            agent_result = analyze_image_dataset(dataset_path, dataset_id)
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

    # ✅ Fonction améliorée pour JSON safe
    def make_json_safe(obj):
        """Convertit en JSON-safe ET nettoie NaN/Infinity."""
        import numpy as np
        
        if isinstance(obj, pd.DataFrame):
            # Remplace NaN et inf AVANT conversion
            obj = obj.fillna(0).replace([float('inf'), float('-inf')], 0)
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            obj = obj.fillna(0).replace([float('inf'), float('-inf')], 0)
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            # Nettoie les arrays numpy
            obj = np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
            return obj.tolist()
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, (int, str, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_json_safe(v) for v in obj]
        return str(obj)

    # ✅ Nettoie le résultat de l'agent
    agent_result_safe = make_json_safe(agent_result)
    
    # ✅ Double sécurité avec clean_nan_inf
    agent_result_safe = clean_nan_inf(agent_result_safe)
    
    # Sauvegarde dans la DB
    dataset.analysis_json = agent_result_safe
    dataset.save()

    # ✅ Utilise json.dumps avec l'encodeur sûr
    try:
        agent_result_json = json.dumps(agent_result_safe, cls=SafeJSONEncoder)
    except Exception as e:
        print(f"[ERROR] JSON encoding failed: {e}")
        # Fallback : force tout en string
        agent_result_json = json.dumps({"error": "Encoding failed", "message": str(e)})

    return render(request, "core/dashboard_bi.html", {
        "agent2_result_json": agent_result_json,
        "dataset_name": dataset.name,
        "dataset_type": dataset_type,
        "preview": preview,
        "dataset": dataset,
    })


# ------------------------
# Chatbot view
# ------------------------
@csrf_exempt
@login_required
def chatbot_view(request, dataset_id):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)

    question = request.POST.get("question", "").strip()
    if not question:
        return JsonResponse({"answer": "Veuillez poser une question."})

    if dataset_id in llm_results_cache:
        answer = llm_results_cache.pop(dataset_id)
        return JsonResponse({"answer": answer})

    thread = threading.Thread(target=process_llm, args=(dataset_id, question), daemon=True)
    thread.start()

    return JsonResponse({"answer": "⏳ Votre question est en cours de traitement. Réessayez dans quelques secondes."})


# ------------------------
# Page loading
# ------------------------
@login_required
def loading_view(request, dataset_id):
    check_status_url = f"/check_status/{dataset_id}/"
    return render(request, "core/loading.html", {"loading_url": check_status_url})