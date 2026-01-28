# datasets/views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from .forms import DatasetUploadForm
from .models import Dataset
import threading
import json

# Importer les agents
from agents.agent2 import analyze_csv_dataset, analyze_image_dataset

# Dictionnaire global temporaire pour stocker les résultats d'analyse
ANALYSIS_RESULTS = {}


def run_analysis(dataset_path, filename, dataset_id):
    """
    Lancer l'analyse CSV ou ZIP d'images en arrière-plan.
    Stocke le résultat dans ANALYSIS_RESULTS.
    """
    global ANALYSIS_RESULTS
    try:
        if filename.endswith('.csv'):
            result = analyze_csv_dataset(dataset_path)
        elif filename.endswith('.zip'):
            result = analyze_image_dataset(dataset_path)
        else:
            result = {"status": "error", "message": "Seuls CSV ou ZIP d'images sont supportés"}
    except Exception as e:
        result = {"status": "error", "message": str(e)}

    ANALYSIS_RESULTS[dataset_id] = result


# ------------------------
# Upload d'un dataset
# ------------------------
@login_required
def dataset_upload_view(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.user = request.user
            dataset.save()

            dataset_id = dataset.id
            dataset_path = dataset.file.path
            filename = dataset.file.name.lower()

            # Lancer l'analyse en arrière-plan
            thread = threading.Thread(target=run_analysis, args=(dataset_path, filename, dataset_id), daemon=True)
            thread.start()

            # URLs pour le loading
            check_url = reverse('core:check_status', kwargs={'dataset_id': dataset_id})
            dashboard_url = reverse('core:dashboard_bi', kwargs={'dataset_id': dataset_id})

            # Rediriger vers la page de loading
            return render(request, "core/loading.html", {
                "check_url": check_url,
                "dashboard_url": dashboard_url
            })
        else:
            return JsonResponse({"status": "error", "errors": form.errors}, status=400)

    form = DatasetUploadForm()
    return render(request, 'datasets/upload_form.html', {'form': form})


# ------------------------
# Vérification AJAX de l'analyse
# ------------------------
@login_required
def check_analysis_status(request, dataset_id):
    result = ANALYSIS_RESULTS.get(dataset_id)
    if result:
        return JsonResponse({"status": "done"})
    return JsonResponse({"status": "pending"})


# ------------------------
# Dashboard BI
# ------------------------
@login_required
def dashboard_view(request, dataset_id):
    result = ANALYSIS_RESULTS.get(dataset_id)
    dataset = Dataset.objects.get(id=dataset_id)

    # Si le résultat n'existe plus, relancer l'analyse
    if not result:
        dataset_path = dataset.file.path
        filename = dataset.file.name.lower()
        if filename.endswith('.csv'):
            result = analyze_csv_dataset(dataset_path)
        else:
            result = analyze_image_dataset(dataset_path)
        ANALYSIS_RESULTS[dataset_id] = result

    return render(request, "core/dashboard_bi.html", {
        "agent2_result_json": json.dumps(result),
        "dataset_name": dataset.name
    })


# ------------------------
# Lister les datasets (JSON)
# ------------------------
@login_required
def dataset_list_view(request):
    datasets = Dataset.objects.filter(user=request.user).values('id', 'name', 'uploaded_at')
    return JsonResponse(list(datasets), safe=False)
