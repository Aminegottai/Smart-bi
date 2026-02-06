import json
import os
import threading

from dotenv import load_dotenv

from core.models import Dataset

# Charge .env (doit contenir: GROQ_API_KEY=...)
# Place le fichier .env à la racine du projet (là où tu lances Django/manage.py)
load_dotenv()

# Cache temporaire pour stocker les réponses LLM
llm_results_cache = {}


# ------------------------
# Fonctions utilitaires
# ------------------------
def make_json_safe(obj):
    """Convertit numpy/pandas en types Python sérialisables (list/dict/str/int...)."""
    import numpy as np
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return str(obj)


def summarize_json_part(json_part, max_keys=10):
    """Résume une partie du JSON pour éviter trop de tokens."""
    json_part = make_json_safe(json_part)

    if isinstance(json_part, dict):
        return {k: json_part[k] for i, k in enumerate(json_part) if i < max_keys}
    if isinstance(json_part, list):
        return json_part[:max_keys]
    return json_part


def chunk_json(json_obj, max_keys_per_chunk=500):
    """Découpe le JSON en chunks plus petits (si besoin)."""
    json_obj = make_json_safe(json_obj)

    chunks = []
    if isinstance(json_obj, dict):
        keys = list(json_obj.keys())
        for i in range(0, len(keys), max_keys_per_chunk):
            chunk = {k: json_obj[k] for k in keys[i : i + max_keys_per_chunk]}
            chunks.append(make_json_safe(chunk))
    elif isinstance(json_obj, list):
        for i in range(0, len(json_obj), max_keys_per_chunk):
            chunk = json_obj[i : i + max_keys_per_chunk]
            chunks.append(make_json_safe(chunk))
    else:
        chunks.append(make_json_safe(json_obj))
    return chunks


# ------------------------
# Résumés texte (graphes + clustering) pour Groq (texte-only)
# ------------------------
def extract_graph_summaries_text(analysis_json_safe, max_chars=6000):
    """
    Transforme analysis_json['graph_summaries'] en texte.
    Groq ne lit pas les images, donc on envoie ce résumé (stats/tendances/corrélations).
    """
    if not isinstance(analysis_json_safe, dict):
        return ""

    gs = analysis_json_safe.get("graph_summaries")
    if not gs:
        return ""

    text = "=== GRAPH SUMMARIES (from plots) ===\n"
    text += json.dumps(gs, ensure_ascii=False, indent=2)
    return text[:max_chars]


def extract_clustering_text(analysis_json_safe, max_chars=2500):
    """
    Envoie explicitement le bloc clustering au LLM
    (cluster_sizes, silhouette_score, PCA variance, inertie, etc.)
    """
    if not isinstance(analysis_json_safe, dict):
        return ""

    clustering = analysis_json_safe.get("clustering")
    if not clustering:
        return ""

    text = "=== CLUSTERING (KMeans + PCA) ===\n"
    text += json.dumps(clustering, ensure_ascii=False, indent=2)
    return text[:max_chars]


# ------------------------
# Construction du prompt
# ------------------------
def build_analysis_context(json_chunk, dataset_type="csv"):
    """
    Prépare le contexte pour le LLM à partir d'un JSON chunk.
    dataset_type: 'csv' ou 'image'
    """
    context = f"Dataset type: {dataset_type}\n"
    context += json.dumps(make_json_safe(json_chunk), indent=2, ensure_ascii=False)
    return context


def build_prompt(context, question):
    """Construit le prompt final pour LLM."""
    return f"{context}\n\nQuestion: {question}\nRéponds clairement et succinctement."


# ------------------------
# Appel au modèle LLM via GROQ API (open-source models)
# ------------------------
def ask_groq(
    prompt,
    model="llama-3.1-8b-instant",
    max_tokens=500,
    temperature=0.0,
):
    """
    Appel LLM via Groq API.
    Nécessite: GROQ_API_KEY (via .env ou variables d'environnement).
    """
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "❌ GROQ_API_KEY manquante (mets-la dans .env ou dans les variables d'environnement)."

    client = Groq(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Réponds clairement et succinctement."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ------------------------
# Traitement LLM dans un thread (avec RAG)
# ------------------------
def process_llm(dataset_id, question):
    """Thread pour traiter la question via LLM et stocker la réponse dans le cache (RAG + graph summaries + clustering)."""
    try:
        from core.rag import RAGIndex

        dataset = Dataset.objects.get(id=dataset_id)
        if not dataset.analysis_json:
            llm_results_cache[dataset_id] = "Analyse non disponible."
            return

        # Convertir JSON string en objet Python si nécessaire
        if isinstance(dataset.analysis_json, str):
            analysis_json = json.loads(dataset.analysis_json)
        else:
            analysis_json = dataset.analysis_json

        analysis_json_safe = make_json_safe(analysis_json)

        dataset_type = "csv" if dataset.file.name.lower().endswith(".csv") else "image"

        # ✅ Toujours injecter ces résumés dans le prompt
        graphs_text = extract_graph_summaries_text(analysis_json_safe)
        clustering_text = extract_clustering_text(analysis_json_safe)

        # RAG: sélectionner les chunks pertinents
        rag = RAGIndex()
        rag.build(analysis_json_safe)
        top_chunks = rag.query(question, top_k=2)

        # Si RAG ne retourne rien, on tente quand même avec graphs_text/clustering_text
        if not top_chunks:
            base_context = f"Dataset type: {dataset_type}\n"
            if graphs_text:
                base_context += graphs_text + "\n\n"
            if clustering_text:
                base_context += clustering_text + "\n\n"

            llm_results_cache[dataset_id] = ask_groq(build_prompt(base_context, question))
            return

        partial_answers = []
        for chunk_text in top_chunks:
            chunk_text = chunk_text[:8000]

            context = f"Dataset type: {dataset_type}\n"
            if graphs_text:
                context += graphs_text + "\n\n"
            if clustering_text:
                context += clustering_text + "\n\n"
            context += chunk_text

            prompt = build_prompt(context, question)
            answer = ask_groq(prompt)
            partial_answers.append(answer)

        llm_results_cache[dataset_id] = "\n---\n".join(partial_answers)

    except Exception as e:
        llm_results_cache[dataset_id] = f"Erreur IA : {str(e)}"


def start_llm_thread(dataset_id, question):
    """Lance le traitement LLM dans un thread."""
    thread = threading.Thread(target=process_llm, args=(dataset_id, question))
    thread.daemon = True
    thread.start()
    return thread