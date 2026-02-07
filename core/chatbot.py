import json
import os
import threading

from dotenv import load_dotenv

from core.models import Dataset

load_dotenv()

llm_results_cache = {}

# ✅ Limite stricte pour Groq
MAX_PROMPT_CHARS = 10000  # ~2500 tokens (4 chars ≈ 1 token)
MAX_RESPONSE_TOKENS = 400  # Réduit pour laisser plus de place au prompt


# ------------------------
# Fonctions utilitaires
# ------------------------
def make_json_safe(obj):
    """Convertit numpy/pandas en types Python sérialisables."""
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


def truncate_text(text, max_chars):
    """Tronque le texte intelligemment."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


# ------------------------
# Résumés compacts
# ------------------------
def extract_graph_summaries_text(analysis_json_safe, max_chars=2000):
    """Extrait résumés de graphes (version compacte)."""
    if not isinstance(analysis_json_safe, dict):
        return ""

    gs = analysis_json_safe.get("graph_summaries")
    if not gs:
        return ""

    text = "=== GRAPHIQUES ===\n"
    text += json.dumps(gs, ensure_ascii=False, indent=1)
    return truncate_text(text, max_chars)


def extract_clustering_text(analysis_json_safe, max_chars=1500):
    """Extrait infos clustering (version compacte)."""
    if not isinstance(analysis_json_safe, dict):
        return ""

    clustering = analysis_json_safe.get("clustering")
    if not clustering:
        return ""

    text = "=== CLUSTERING ===\n"
    text += json.dumps(clustering, ensure_ascii=False, indent=1)
    return truncate_text(text, max_chars)


# ------------------------
# Construction du prompt (avec limite stricte)
# ------------------------
def build_context(json_chunk, dataset_type, graphs_text, clustering_text, max_chars):
    """Construit le contexte en respectant la limite de caractères."""
    context = f"Type: {dataset_type}\n\n"
    
    remaining = max_chars - len(context) - 200  # Réserve pour question
    
    # Priorise graphiques et clustering
    if graphs_text:
        allocated = min(len(graphs_text), remaining // 3)
        context += truncate_text(graphs_text, allocated) + "\n\n"
        remaining -= allocated
    
    if clustering_text:
        allocated = min(len(clustering_text), remaining // 3)
        context += truncate_text(clustering_text, allocated) + "\n\n"
        remaining -= allocated
    
    # Ajoute chunk data
    if json_chunk and remaining > 500:
        chunk_str = json.dumps(make_json_safe(json_chunk), indent=1, ensure_ascii=False)
        context += "=== DONNÉES ===\n"
        context += truncate_text(chunk_str, remaining)
    
    return context


def build_prompt(context, question):
    """Construit le prompt final avec limite stricte."""
    full_prompt = f"{context}\n\nQuestion: {question}\nRéponds de façon claire et concise."
    
    # ✅ Sécurité : tronque si trop long
    if len(full_prompt) > MAX_PROMPT_CHARS:
        full_prompt = truncate_text(full_prompt, MAX_PROMPT_CHARS - 100)
        full_prompt += f"\n\nQuestion: {question}\nRéponds brièvement."
    
    return full_prompt


# ------------------------
# Appel Groq avec gestion erreurs
# ------------------------
def ask_groq(
    prompt,
    model="llama-3.1-8b-instant",
    max_tokens=MAX_RESPONSE_TOKENS,
    temperature=0.0,
):
    """Appel LLM via Groq avec retry sur rate limit."""
    from groq import Groq
    import time

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "❌ GROQ_API_KEY manquante."

    # ✅ Vérifie taille du prompt (sécurité)
    estimated_tokens = len(prompt) // 4
    if estimated_tokens + max_tokens > 5500:  # Marge de sécurité
        max_tokens = max(100, 5500 - estimated_tokens)
        prompt = truncate_text(prompt, MAX_PROMPT_CHARS)

    client = Groq(api_key=api_key)

    try:
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
    
    except Exception as e:
        error_msg = str(e)
        
        # ✅ Si rate limit, réessaye avec prompt plus court
        if "413" in error_msg or "rate_limit" in error_msg.lower():
            print(f"⚠️ Rate limit hit, réduction du prompt...")
            shorter_prompt = truncate_text(prompt, MAX_PROMPT_CHARS // 2)
            
            try:
                time.sleep(1)  # Petite pause
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Réponds brièvement."},
                        {"role": "user", "content": shorter_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=200,  # Réduit drastiquement
                )
                return resp.choices[0].message.content.strip()
            except:
                return "⚠️ Prompt trop long, impossible de répondre. Pose une question plus ciblée."
        
        return f"❌ Erreur API: {error_msg}"


# ------------------------
# Traitement LLM avec RAG
# ------------------------
def process_llm(dataset_id, question):
    try:
        from core.rag import RAGIndex

        dataset = Dataset.objects.get(id=dataset_id)
        if not dataset.analysis_json:
            llm_results_cache[dataset_id] = "Analyse non disponible."
            return

        if isinstance(dataset.analysis_json, str):
            analysis_json = json.loads(dataset.analysis_json)
        else:
            analysis_json = dataset.analysis_json

        analysis_json_safe = make_json_safe(analysis_json)
        dataset_type = "csv" if dataset.file.name.lower().endswith(".csv") else "image"

        # ✅ Résumés compacts
        graphs_text = extract_graph_summaries_text(analysis_json_safe, max_chars=1500)
        clustering_text = extract_clustering_text(analysis_json_safe, max_chars=1000)

        # RAG
        rag = RAGIndex()
        rag.build(analysis_json_safe)
        top_chunks = rag.query(question, top_k=2)

        if not top_chunks:
            # Contexte minimal
            context = build_context(
                json_chunk=None,
                dataset_type=dataset_type,
                graphs_text=graphs_text,
                clustering_text=clustering_text,
                max_chars=MAX_PROMPT_CHARS - 500
            )
            prompt = build_prompt(context, question)
            llm_results_cache[dataset_id] = ask_groq(prompt)
            return

        # Traite chaque chunk séparément
        partial_answers = []
        for chunk_text in top_chunks[:2]:  # Max 2 chunks
            context = build_context(
                json_chunk=chunk_text[:3000],  # Limite chunk
                dataset_type=dataset_type,
                graphs_text=graphs_text,
                clustering_text=clustering_text,
                max_chars=MAX_PROMPT_CHARS - 500
            )
            
            prompt = build_prompt(context, question)
            answer = ask_groq(prompt)
            partial_answers.append(answer)

        llm_results_cache[dataset_id] = "\n---\n".join(partial_answers)

    except Exception as e:
        llm_results_cache[dataset_id] = f"Erreur IA : {str(e)}"


def start_llm_thread(dataset_id, question):
    thread = threading.Thread(target=process_llm, args=(dataset_id, question))
    thread.daemon = True
    thread.start()
    return thread