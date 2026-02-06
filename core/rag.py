import json
from typing import List, Union, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------
# Fonctions utilitaires
# ------------------------
def make_json_safe(obj: Any) -> Any:
    """
    Convertit des objets non sérialisables (numpy/pandas) en structures Python simples.
    """
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


def json_to_text(json_obj: Any, max_keys_per_item: int = 10) -> str:
    """
    Transforme un JSON (dict/list) en texte lisible pour le RAG/LLM.
    - max_keys_per_item: limite les clés par item pour réduire la taille.
    """
    json_obj = make_json_safe(json_obj)

    if isinstance(json_obj, dict):
        keys = list(json_obj.keys())[:max_keys_per_item]
        return "\n".join(
            f"{k}: {json.dumps(make_json_safe(json_obj[k]), ensure_ascii=False)}"
            for k in keys
        )

    if isinstance(json_obj, list):
        # Convertit chaque item en texte et les sépare par une ligne
        parts = [json_to_text(item, max_keys_per_item=max_keys_per_item) for item in json_obj[:max_keys_per_item]]
        return "\n---\n".join(parts)

    return str(json_obj)


def chunk_json(json_obj: Any, max_items_per_chunk: int = 50) -> List[Any]:
    """
    Découpe le JSON en chunks pour l'index RAG.
    - Si list: chunks de max_items_per_chunk items
    - Si dict: convertit en liste de {k: v} puis chunks
    """
    json_obj = make_json_safe(json_obj)

    chunks: List[Any] = []

    if isinstance(json_obj, list):
        for i in range(0, len(json_obj), max_items_per_chunk):
            chunks.append(json_obj[i:i + max_items_per_chunk])
        return chunks

    if isinstance(json_obj, dict):
        items = [{k: json_obj[k]} for k in list(json_obj.keys())]
        for i in range(0, len(items), max_items_per_chunk):
            chunks.append(items[i:i + max_items_per_chunk])
        return chunks

    # fallback: un seul chunk
    return [[json_obj]]


# ------------------------
# Création de l'index TF-IDF pour RAG
# ------------------------
class RAGIndex:
    """
    Index simple basé sur TF-IDF pour Retrieval-Augmented Generation.
    """
    def __init__(self):
        self.text_chunks: List[str] = []
        self.vectorizer: Union[TfidfVectorizer, None] = None
        self.vectors = None  # matrice sparse

    def build(self, json_obj: Any, max_items_per_chunk: int = 50, max_keys_per_item: int = 10) -> None:
        """
        Construit l'index TF-IDF à partir d'un JSON.
        """
        json_obj_safe = make_json_safe(json_obj)
        chunks = chunk_json(json_obj_safe, max_items_per_chunk=max_items_per_chunk)

        self.text_chunks = [json_to_text(chunk, max_keys_per_item=max_keys_per_item) for chunk in chunks]

        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(self.text_chunks)

    def query(self, question: str, top_k: int = 3) -> List[str]:
        """
        Retourne les top_k chunks les plus pertinents pour la question.
        """
        if self.vectorizer is None or self.vectors is None:
            return []

        # index vide
        if getattr(self.vectors, "shape", (0, 0))[0] == 0:
            return []

        question_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vec, self.vectors)[0]

        top_indices = scores.argsort()[::-1][:top_k]
        return [self.text_chunks[i] for i in top_indices]


# ------------------------
# Exemple d'utilisation (sans question statique)
# ------------------------
if __name__ == "__main__":
    # JSON fictif
    example_json = [
        {"id": 1, "name": "Alice", "age": 30, "city": "Paris"},
        {"id": 2, "name": "Bob", "age": 25, "city": "London"},
        {"id": 3, "name": "Charlie", "age": 28, "city": "Berlin"},
    ]

    rag_index = RAGIndex()
    rag_index.build(example_json)

    # Demander la question à l'utilisateur (pas statique)
    question = input("Entrez votre question: ").strip()
    if question:
        relevant_chunks = rag_index.query(question, top_k=3)
        print("\nChunks pertinents :")
        for chunk in relevant_chunks:
            print(chunk)
            print("===")
    else:
        print("Aucune question fournie.")