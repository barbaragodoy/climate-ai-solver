"""Módulo de agrupamento K-Means para perfis climáticos sazonais."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import joblib

FEAT_CLUSTER = [
    'temp_media', 'temp_max', 'temp_min',
    'precipitacao', 'umidade_relativa',
    'velocidade_vento', 'radiacao_solar', 'amplitude_termica'
]

CLUSTER_NAMES = {
    0: 'Chuvoso-Quente',
    1: 'Seco-Crítico',
    2: 'Seco-Ameno',
    3: 'Transição',
}


def find_optimal_k(X_scaled: np.ndarray, k_max: int = 9) -> dict:
    results = {}
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        results[k] = {
            'inertia': km.inertia_,
            'silhouette': silhouette_score(X_scaled, labels, sample_size=5000, random_state=42),
        }
    return results


def train(df: pd.DataFrame, n_clusters: int = 4):
    df_m = df[FEAT_CLUSTER].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_m)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    labels = model.fit_predict(X_scaled)

    df_m = df_m.copy()
    df_m['cluster'] = labels
    df_m['cluster_nome'] = df_m['cluster'].map(CLUSTER_NAMES)

    sil = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
    db = davies_bouldin_score(X_scaled, labels)

    centroids = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=FEAT_CLUSTER
    )

    metrics = {'silhouette': sil, 'davies_bouldin': db}
    return model, scaler, df_m, centroids, metrics


def predict_cluster(model, scaler, input_df: pd.DataFrame) -> np.ndarray:
    X = scaler.transform(input_df[FEAT_CLUSTER])
    return model.predict(X)


def save(model, scaler, model_path: str, scaler_path: str) -> None:
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
