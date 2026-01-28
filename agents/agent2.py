from math import sqrt
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

GRAPH_DIR = Path("media/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)



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

    return {
        "numeric": num_cols,
        "categorical": cat_cols,
        "date": date_cols,
    }


def _iqr_outliers(df: pd.DataFrame, num_cols):
    outliers = {}
    for col in num_cols:
        series = df[col].dropna()
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
    if len(num_cols) < 2:
        return []
    corr = df[num_cols].corr().abs()
    for i in range(len(num_cols)):
        corr.iat[i, i] = 0
    pairs = (
        corr.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "corr"})
    )
    pairs = pairs.sort_values("corr", ascending=False).drop_duplicates(subset=["corr", "feature_1", "feature_2"])
    return pairs.head(top_n).to_dict(orient="records")


def _generate_graphs(df: pd.DataFrame, types: dict, dataset_name: str, max_graphs_per_type: int = 8):
    graphs = []
    for col in types["numeric"][:max_graphs_per_type]:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution de {col}")
        path = GRAPH_DIR / f"{dataset_name}_{col}_hist.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        graphs.append(f"/media/graphs/{path.name}")

    for col in types["numeric"][:max_graphs_per_type]:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot de {col}")
        path = GRAPH_DIR / f"{dataset_name}_{col}_box.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        graphs.append(f"/media/graphs/{path.name}")

    for col in types["categorical"][:max_graphs_per_type]:
        plt.figure(figsize=(8, 4))
        df[col].value_counts().head(15).plot(kind="bar")
        plt.title(f"Top catégories de {col}")
        plt.xticks(rotation=30, ha="right")
        path = GRAPH_DIR / f"{dataset_name}_{col}_bar.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        graphs.append(f"/media/graphs/{path.name}")

    if len(types["numeric"]) >= 2:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[types["numeric"]].corr(), annot=False, cmap="coolwarm")
        plt.title("Matrice de corrélation")
        path = GRAPH_DIR / f"{dataset_name}_correlation_matrix.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        graphs.append(f"/media/graphs/{path.name}")

    if types["date"] and types["numeric"]:
        date_col = types["date"][0]
        date_series = pd.to_datetime(df[date_col], errors="coerce",dayfirst=True)
        for num_col in types["numeric"][:max_graphs_per_type]:
            plt.figure(figsize=(10, 4))
            tmp = df[[num_col]].copy()
            tmp[date_col] = date_series
            tmp = tmp.dropna(subset=[date_col]).sort_values(date_col)
            tmp.groupby(date_col)[num_col].mean().plot()
            plt.title(f"Tendance temporelle de {num_col} (moyenne)")
            plt.xticks(rotation=30, ha="right")
            path = GRAPH_DIR / f"{dataset_name}_{num_col}_trend.png"
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            graphs.append(f"/media/graphs/{path.name}")

    return graphs


def _run_clustering(df: pd.DataFrame, types: dict):
    num_cols = types["numeric"]
    if len(num_cols) < 2 or len(df) < 3:
        return {"status": "skipped", "reason": "Pas assez de colonnes numériques pour le clustering."}

    X = df[num_cols].fillna(0)
    n_clusters = min(4, len(X))
    if n_clusters < 2:
        return {"status": "skipped", "reason": "Pas assez d'observations pour le clustering."}

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis")
    plt.title("Clusters PCA")
    path = GRAPH_DIR / "pca_clusters.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return {
        "status": "success",
        "n_clusters": n_clusters,
        "cluster_labels": clusters.tolist(),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "scatter_plot": f"/media/graphs/{path.name}",
    }


# Remplace la fonction _parse_dates par :
def _parse_dates(series: pd.Series):
    """
    Convertit une série en datetime sans générer de warnings et compatible avec toutes les versions de pandas.
    """
    try:
        return pd.to_datetime(series, errors="coerce",dayfirst=True)
    except Exception:
        # fallback très strict
        return pd.to_datetime(series.astype(str), errors="coerce",dayfirst=True)



import pandas as pd

def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime toutes les colonnes dont le nom contient 'id' (case-insensitive).
    """
    cols_to_drop = [col for col in df.columns if 'id' in col.lower()]
    return df.drop(columns=cols_to_drop)



def analyze_csv_dataset(dataset_path: str, column_roles: dict | None = None, max_graphs_per_type: int = 8):
    """
    Agent 2 : EDA + ML + visualisations.
    - S'exécute après récupération des rôles de colonnes par agent1.
    - Produit stats descriptives, graphes, corrélations, clustering.
    """


    df = pd.read_csv(dataset_path)
    df = drop_id_columns(df)
    dataset_name = Path(dataset_path).stem


    types = _detect_types(df)
    date_cols = [col for col in types["categorical"] 
             if pd.api.types.is_datetime64_any_dtype(df[col]) 
             or pd.to_datetime(df[col], errors='coerce').notna().all()]


    result = {
        "status": "success",
        "shape": df.shape,
        "columns": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "types": types,
        "numeric_summary": df.select_dtypes(include="number").describe().to_dict(),
        "categorical_summary": {
    col: df[col].value_counts().to_dict()
    for col in types["categorical"]
    if col not in date_cols  # <-- on utilise date_cols détecté automatiquement
},
    }

    result["outliers"] = _iqr_outliers(df, types["numeric"])
    result["correlations"] = _top_correlations(df, types["numeric"])
    result["visualizations"] = _generate_graphs(df, types, dataset_name, max_graphs_per_type)
    result["clustering"] = _run_clustering(df, types)

    return result
from PIL import Image
import numpy as np
import random

def analyze_image_dataset(image_dir: str, max_images: int = 5):
    """
    Analyse un dataset d'images :
    - Nombre d'images par classe (dossier)
    - Taille moyenne des images
    - Affichage aléatoire de quelques images
    - Histogrammes de pixels (grayscale)
    """
    image_dir = Path(image_dir)
    classes = [d.name for d in image_dir.iterdir() if d.is_dir()]
    stats = {"num_classes": len(classes), "classes": {}}

    for cls in classes:
        cls_path = image_dir / cls
        images = list(cls_path.glob("*.[pj][pn]g"))  # jpg/png
        stats["classes"][cls] = {"num_images": len(images)}
        widths, heights = [], []
        for img_path in images:
            try:
                img = Image.open(img_path)
                widths.append(img.width)
                heights.append(img.height)
            except Exception:
                continue
        if widths:
            stats["classes"][cls]["avg_width"] = int(np.mean(widths))
            stats["classes"][cls]["avg_height"] = int(np.mean(heights))

    # Affichage aléatoire
    random_imgs = []
    for cls in classes:
        cls_path = image_dir / cls
        imgs = list(cls_path.glob("*.[pj][pn]g"))
        random_imgs += random.sample(imgs, min(max_images, len(imgs)))

    for i, img_path in enumerate(random_imgs):
        img = Image.open(img_path)
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{img_path.parent.name} - {img_path.name}")
        path = GRAPH_DIR / f"image_preview_{i}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        stats[f"preview_{i}"] = f"/media/graphs/{path.name}"

    # Histogrammes de pixels (grayscale)
    for i, img_path in enumerate(random_imgs):
        try:
            img = Image.open(img_path).convert("L")  # convert to grayscale
            arr = np.array(img)
            plt.figure()
            plt.hist(arr.flatten(), bins=256, color="gray")
            plt.title(f"Histogramme pixels - {img_path.name}")
            path = GRAPH_DIR / f"image_hist_{i}.png"
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            stats[f"hist_{i}"] = f"/media/graphs/{path.name}"
        except Exception:
            continue

    return stats
