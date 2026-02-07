from math import sqrt
from pathlib import Path
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

GRAPH_DIR = Path("media/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def _parse_dates(series: pd.Series):
    """
    Convertit une série en datetime sans générer de warnings.
    """
    try:
        return pd.to_datetime(series, errors="coerce", dayfirst=True)
    except Exception:
        return pd.to_datetime(series.astype(str), errors="coerce", dayfirst=True)


def _detect_types(df: pd.DataFrame):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    date_cols = []

    for col in df.columns:
        if col in num_cols:
            continue

        parsed = _parse_dates(df[col])
        if parsed.notna().mean() >= 0.6:
            date_cols.append(col)
            if col in cat_cols:
                cat_cols.remove(col)

    return {"numeric": num_cols, "categorical": cat_cols, "date": date_cols}


def get_graph_dir_by_id(dataset_id: int) -> Path:
    graph_dir = Path("media/graphs") / f"dataset_{dataset_id}"
    graph_dir.mkdir(parents=True, exist_ok=True)
    return graph_dir


def _iqr_outliers(df: pd.DataFrame, num_cols):
    outliers = {}
    for col in num_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            outliers[col] = {"count": 0, "ratio": 0.0}
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        outliers[col] = {"count": int(mask.sum()), "ratio": float(mask.mean())}
    return outliers


def _top_correlations(df: pd.DataFrame, num_cols, top_n: int = 15):
    """
    Retourne les top N paires de colonnes les plus corrélées.
    """
    if len(num_cols) < 2:
        return []
    corr = df[num_cols].corr(numeric_only=True).abs()
    for i in range(len(num_cols)):
        corr.iat[i, i] = 0
    pairs = (
        corr.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "corr"})
    )
    pairs = pairs.sort_values("corr", ascending=False)
    return pairs.head(top_n).to_dict(orient="records")


# ✅ NOUVELLE FONCTION: Matrice de corrélation complète JSON-safe
def _full_correlation_matrix(df: pd.DataFrame, num_cols) -> dict:
    """
    Retourne la matrice de corrélation complète en format JSON-safe.
    """
    if len(num_cols) < 2:
        return {}
    
    corr = df[num_cols].corr(numeric_only=True)
    
    # Convertit en dict avec valeurs float Python (pas numpy)
    corr_dict = {}
    for col in corr.columns:
        corr_dict[col] = {
            row: float(corr.loc[row, col]) 
            for row in corr.index
        }
    
    return corr_dict


def _generate_graphs(
    df: pd.DataFrame,
    types: dict,
    dataset_name: str,
    dataset_id: int,
    max_graphs_per_type: int = 8,
):
    """
    Génère les graphes (png) et retourne la liste des paths.
    NOTE: on garde la limite max_graphs_per_type pour éviter trop d'images.
    """
    graph_dir = get_graph_dir_by_id(dataset_id)
    graphs = []

    for col in types["numeric"][:max_graphs_per_type]:
        plt.figure()
        sns.histplot(pd.to_numeric(df[col], errors="coerce").dropna(), kde=True)
        plt.title(f"Distribution de {col}")
        path = graph_dir / f"{dataset_name}_{col}_hist.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        graphs.append(f"/media/graphs/dataset_{dataset_id}/{path.name}")

    for col in types["numeric"][:max_graphs_per_type]:
        plt.figure()
        sns.boxplot(x=pd.to_numeric(df[col], errors="coerce"))
        plt.title(f"Boxplot de {col}")
        path = graph_dir / f"{dataset_name}_{col}_box.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        graphs.append(f"/media/graphs/dataset_{dataset_id}/{path.name}")

    for col in types["categorical"][:max_graphs_per_type]:
        plt.figure(figsize=(8, 4))
        df[col].value_counts().head(15).plot(kind="bar")
        plt.title(f"Top catégories de {col}")
        plt.xticks(rotation=30, ha="right")
        path = graph_dir / f"{dataset_name}_{col}_bar.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        graphs.append(f"/media/graphs/dataset_{dataset_id}/{path.name}")

    if len(types["numeric"]) >= 2:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[types["numeric"]].corr(numeric_only=True), annot=False, cmap="coolwarm")
        plt.title("Matrice de corrélation")
        path = graph_dir / f"{dataset_name}_correlation_matrix.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        graphs.append(f"/media/graphs/dataset_{dataset_id}/{path.name}")

    if types["date"] and types["numeric"]:
        date_col = types["date"][0]
        date_series = _parse_dates(df[date_col])

        for num_col in types["numeric"][:max_graphs_per_type]:
            plt.figure(figsize=(10, 4))
            tmp = df[[num_col]].copy()
            tmp[date_col] = date_series
            tmp[num_col] = pd.to_numeric(tmp[num_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col]).sort_values(date_col)

            tmp.groupby(date_col)[num_col].mean().plot()
            plt.title(f"Tendance temporelle de {num_col}")
            path = graph_dir / f"{dataset_name}_{num_col}_trend.png"
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            graphs.append(f"/media/graphs/dataset_{dataset_id}/{path.name}")

    return graphs


def _run_clustering(df: pd.DataFrame, types: dict, dataset_id: int):
    """
    Clustering KMeans + projection PCA 2D + métriques (silhouette, inertie)
    IMPORTANT: toutes les valeurs sont converties en types JSON-safe.
    """
    num_cols = types["numeric"]
    if len(num_cols) < 2 or len(df) < 3:
        return {"status": "skipped", "reason": "Pas assez de colonnes numériques pour le clustering."}

    X = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    n_clusters = min(4, len(X))
    if n_clusters < 2:
        return {"status": "skipped", "reason": "Pas assez d'observations pour le clustering."}

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    sil = None
    if len(set(clusters)) > 1 and len(X) > n_clusters:
        sil = float(silhouette_score(X, clusters))

    counts = Counter(clusters)
    cluster_sizes = {int(k): int(v) for k, v in counts.items()}

    graph_dir = get_graph_dir_by_id(dataset_id)
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis")
    plt.title("Clusters PCA")
    path = graph_dir / "pca_clusters.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return {
        "status": "success",
        "n_clusters": int(n_clusters),
        "cluster_labels": [int(x) for x in clusters.tolist()],
        "cluster_sizes": cluster_sizes,
        "pca_explained_variance": [float(x) for x in pca.explained_variance_ratio_.tolist()],
        "kmeans_inertia": float(kmeans.inertia_),
        "silhouette_score": float(sil) if sil is not None else None,
        "scatter_plot": f"/media/graphs/dataset_{dataset_id}/{path.name}",
    }


def _clustering_summary(clustering: dict):
    """Résumé du clustering pour le LLM."""
    if not isinstance(clustering, dict) or clustering.get("status") != "success":
        return {"status": "skipped"}

    cluster_sizes = clustering.get("cluster_sizes") or {}
    return {
        "status": "success",
        "n_clusters": clustering.get("n_clusters"),
        "cluster_sizes": cluster_sizes,
        "pca_explained_variance": clustering.get("pca_explained_variance"),
        "silhouette_score": clustering.get("silhouette_score"),
        "kmeans_inertia": clustering.get("kmeans_inertia"),
    }


def _build_graph_summaries_for_generated_graphs(
    df: pd.DataFrame,
    types: dict,
    max_graphs_per_type: int = 8,
    top_k_categories: int = 15,
    top_corr_n: int = 15,
):
    """
    Résume EXACTEMENT les colonnes utilisées pour générer les graphes.
    
    ✅ MODIFIÉ: Inclut maintenant la matrice de corrélation complète
    pour que le LLM puisse répondre aux questions sur les corrélations.
    """
    used_numeric = types.get("numeric", [])[:max_graphs_per_type]
    used_categorical = types.get("categorical", [])[:max_graphs_per_type]

    summaries = {
        "numeric": {},
        "categorical": {},
        "correlations": [],           # ✅ Top corrélations (paires)
        "correlation_matrix": {},     # ✅ NOUVEAU: Matrice complète
        "trends": {},
        "used_columns": {
            "numeric": used_numeric,
            "categorical": used_categorical,
            "date": types.get("date", []),
        },
        "meta": {
            "note": "Text/numeric summaries of generated plots. The LLM does not read PNG files."
        },
    }

    # Résumé des colonnes numériques (pour histogrammes et boxplots)
    for col in used_numeric:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            summaries["numeric"][col] = {"error": "no numeric data"}
            continue

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((s < lower) | (s > upper)).sum())

        summaries["numeric"][col] = {
            "count": int(s.shape[0]),
            "missing": int(df[col].isna().sum()),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
            "q1": q1,
            "q3": q3,
            "iqr": float(iqr),
            "outliers_count": outliers,
        }

    # Résumé des colonnes catégorielles (pour barplots)
    for col in used_categorical:
        s = df[col]
        vc = s.astype(str).value_counts(dropna=False).head(top_k_categories)
        summaries["categorical"][col] = {
            "unique_count": int(s.nunique(dropna=False)),
            "top_values": vc.to_dict(),
        }

    # ✅ Corrélations (top paires + matrice complète)
    if len(used_numeric) >= 2:
        # Top N paires les plus corrélées
        summaries["correlations"] = _top_correlations(df, used_numeric, top_n=top_corr_n)
        
        # ✅ NOUVEAU: Matrice de corrélation complète pour le LLM
        summaries["correlation_matrix"] = _full_correlation_matrix(df, used_numeric)

    # Tendances temporelles
    if types.get("date") and used_numeric:
        date_col = types["date"][0]
        d = df.copy()
        d[date_col] = _parse_dates(d[date_col])

        for col in used_numeric:
            tmp = d[[date_col, col]].copy()
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col, col]).sort_values(date_col)

            if tmp.empty:
                summaries["trends"][col] = {"error": "no data"}
                continue

            g = tmp.groupby(date_col)[col].mean()
            if g.shape[0] < 2:
                summaries["trends"][col] = {"points": int(g.shape[0]), "trend": "insufficient_points"}
                continue

            x = np.arange(g.shape[0], dtype=float)
            y = g.values.astype(float)
            slope = float(np.polyfit(x, y, 1)[0])

            summaries["trends"][col] = {
                "date_col": date_col,
                "points": int(g.shape[0]),
                "start": str(g.index.min()),
                "end": str(g.index.max()),
                "min": float(y.min()),
                "max": float(y.max()),
                "mean": float(y.mean()),
                "slope": slope,
                "trend": "up" if slope > 0 else ("down" if slope < 0 else "flat"),
            }

    return summaries


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime toutes les colonnes dont le nom contient 'id' (case-insensitive)."""
    cols_to_drop = [col for col in df.columns if "id" in col.lower()]
    return df.drop(columns=cols_to_drop)


def analyze_csv_dataset(dataset_path: str, dataset_id: int, column_roles: dict | None = None, max_graphs_per_type: int = 8):
    """
    Analyse complète d'un dataset CSV.
    
    ✅ MODIFIÉ: Inclut la matrice de corrélation dans graph_summaries
    pour que le chatbot LLM puisse répondre aux questions sur les corrélations.
    """
    print("[EDA] Lecture du dataset...")
    df = pd.read_csv(dataset_path)
    df = drop_id_columns(df)
    dataset_name = Path(dataset_path).stem
    print(f"[EDA] Dataset chargé : {dataset_name}, shape : {df.shape}")

    types = _detect_types(df)
    print(f"[EDA] Types détectés : {types}")

    result = {
        "status": "success",
        "shape": df.shape,
        "columns": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "types": types,
        "numeric_summary": df.select_dtypes(include="number").describe().to_dict(),
        "categorical_summary": {col: df[col].value_counts().to_dict() for col in types["categorical"]},
    }

    print("[EDA] Calcul des outliers...")
    result["outliers"] = _iqr_outliers(df, types["numeric"])
    print("[EDA] Outliers calculés.")

    print("[EDA] Calcul des corrélations...")
    result["correlations"] = _top_correlations(df, types["numeric"])
    
    # ✅ NOUVEAU: Matrice de corrélation complète au niveau racine
    result["correlation_matrix"] = _full_correlation_matrix(df, types["numeric"])
    print("[EDA] Corrélations calculées.")

    print("[EDA] Génération des graphes...")
    result["visualizations"] = _generate_graphs(df, types, dataset_name, dataset_id, max_graphs_per_type)
    print(f"[EDA] Graphes générés : {result['visualizations']}")

    # ✅ graph_summaries inclut maintenant correlation_matrix
    result["graph_summaries"] = _build_graph_summaries_for_generated_graphs(
        df,
        types,
        max_graphs_per_type=max_graphs_per_type,
        top_k_categories=15,
        top_corr_n=15,
    )

    print("[EDA] Lancement du clustering...")
    result["clustering"] = _run_clustering(df, types, dataset_id)
    print(f"[EDA] Clustering terminé : {result['clustering']}")

    if isinstance(result.get("clustering"), dict) and result["clustering"].get("status") == "success":
        result["visualizations"].append(result["clustering"]["scatter_plot"])

    result["graph_summaries"]["clustering"] = _clustering_summary(result["clustering"])

    print("[EDA] Analyse terminée.")
    return result


def analyze_image_dataset(zip_or_dir: str, dataset_id: int, max_preview: int = 5):
    """
    Analyse complète d'un dataset d'images pour le dashboard.
    
    ✅ Inclut graph_summaries (texte) pour le chatbot Groq (texte-only).
    """
    from zipfile import ZipFile
    import random
    from PIL import Image

    # Dossier d'extraction
    dataset_path = Path(f"media/datasets_extracted/dataset_{dataset_id}")
    if zip_or_dir.endswith(".zip"):
        with ZipFile(zip_or_dir, "r") as zip_ref:
            zip_ref.extractall(dataset_path)
        print(f"[DEBUG] ZIP extrait vers: {dataset_path}")
    else:
        dataset_path = Path(zip_or_dir)

    # 1) Détecter classes
    classes = [d for d in dataset_path.rglob("*") if d.is_dir() and any(d.glob("*.*"))]
    class_names = [c.name for c in classes]
    stats = {"status": "success", "num_classes": len(classes), "classes": {}}
    print(f"[DEBUG] Classes détectées: {class_names}")

    # 2) Stats + collecte PCA
    all_images = []
    labels = []
    for cls_path in classes:
        cls_name = cls_path.name

        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            images += list(cls_path.glob(ext))
            images += list(cls_path.glob(ext.upper()))

        stats["classes"][cls_name] = {"num_images": len(images)}

        widths, heights = [], []
        for img_path in images:
            try:
                img = Image.open(img_path)
                widths.append(img.width)
                heights.append(img.height)

                all_images.append(np.array(img.convert("RGB").resize((64, 64))).flatten())
                labels.append(cls_name)
            except Exception as e:
                print("[DEBUG] Image open failed:", img_path, e)

        if widths:
            stats["classes"][cls_name]["avg_width"] = int(np.mean(widths))
            stats["classes"][cls_name]["avg_height"] = int(np.mean(heights))

    # 3) Preview
    preview_images = []
    for cls_path in classes:
        imgs = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            imgs += list(cls_path.glob(ext))
            imgs += list(cls_path.glob(ext.upper()))
        if imgs:
            preview_images += random.sample(imgs, min(max_preview, len(imgs)))

    stats.setdefault("visualizations", [])
    graph_dir = Path("media/graphs") / f"dataset_{dataset_id}"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # 3a) Previews (PNG)
    for i, img_path in enumerate(preview_images):
        try:
            img = Image.open(img_path)
            plt.figure()
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{img_path.parent.name} - {img_path.name}")
            save_path = graph_dir / f"image_preview_{i}.png"
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            stats["visualizations"].append({"type": "image_preview", "path": f"/media/graphs/dataset_{dataset_id}/image_preview_{i}.png"})
        except Exception as e:
            print("[DEBUG] Preview failed:", img_path, e)

    # 4) Histogrammes pixels (PNG + résumé)
    pixel_hists = []
    for i, img_path in enumerate(preview_images):
        try:
            img = Image.open(img_path).convert("L")
            arr = np.array(img)

            hist_counts, _ = np.histogram(arr.flatten(), bins=256, range=(0, 255))
            pixel_hists.append(
                {
                    "image": img_path.name,
                    "mean_pixel": float(arr.mean()),
                    "std_pixel": float(arr.std()),
                    "min_pixel": int(arr.min()),
                    "max_pixel": int(arr.max()),
                    # ✅ Optionnel: on peut retirer counts pour réduire la taille
                    # "counts": hist_counts.tolist(),
                }
            )

            plt.figure()
            plt.hist(arr.flatten(), bins=256, color="gray")
            plt.title(f"Histogramme pixels - {img_path.name}")
            save_path = graph_dir / f"image_hist_{i}.png"
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            stats["visualizations"].append({"type": "histogram", "path": f"/media/graphs/dataset_{dataset_id}/image_hist_{i}.png"})
        except Exception as e:
            print("[DEBUG] Histogram failed:", img_path, e)

    # 5) PCA (PNG + résumé)
    pca_summary = {"status": "skipped", "reason": "no images"}
    if all_images:
        try:
            all_images_np = np.array(all_images)
            pca = PCA(n_components=2, random_state=42)
            emb = pca.fit_transform(all_images_np)

            plt.figure(figsize=(6, 6))
            for cls_name in class_names:
                idxs = [i for i, l in enumerate(labels) if l == cls_name]
                if idxs:
                    plt.scatter(emb[idxs, 0], emb[idxs, 1], label=cls_name, s=12)

            plt.title("PCA des images")
            plt.legend()
            save_path = graph_dir / "pca_plot.png"
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            stats["visualizations"].append({"type": "pca", "path": f"/media/graphs/dataset_{dataset_id}/pca_plot.png"})

            pca_summary = {
                "status": "success",
                "n_samples": int(all_images_np.shape[0]),
                "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_.tolist()],
                "classes_included": class_names,
            }
        except Exception as e:
            print("[DEBUG] PCA failed:", e)
            pca_summary = {"status": "failed", "error": str(e)}

    # ✅ graph_summaries pour le chatbot LLM (texte-only)
    stats["graph_summaries"] = {
        "image_dataset": {
            "num_classes": stats["num_classes"],
            "classes": stats["classes"],
            "preview_count": len(preview_images),
        },
        "pixel_histograms_from_previews": pixel_hists,
        "pca": pca_summary,
        "meta": {
            "note": "Text summaries for LLM chatbot. The LLM cannot read PNG files directly."
        },
    }

    return stats