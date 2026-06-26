"""Módulo de classificação: k-NN (onda de calor) e Árvore de Decisão (incêndio)."""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score)
import joblib


FEAT_KNN = ['temp_max', 'temp_min', 'umidade_relativa', 'velocidade_vento',
            'radiacao_solar', 'mes_sin', 'mes_cos', 'temp_media_30d']

FEAT_DT = ['temp_max', 'umidade_relativa', 'velocidade_vento',
           'precipitacao', 'precip_7d', 'radiacao_solar', 'estacao_seca']


def train_knn(df: pd.DataFrame, k: int = 7):
    df_m = df[FEAT_KNN + ['risco_onda_calor']].dropna()
    X, y = df_m[FEAT_KNN], df_m['risco_onda_calor']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='distance')
    model.fit(X_tr_s, y_tr)

    y_pred = model.predict(X_te_s)
    y_prob = model.predict_proba(X_te_s)[:, 1]

    metrics = {
        'report': classification_report(y_te, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_te, y_pred),
        'auc_roc': roc_auc_score(y_te, y_prob),
    }
    return model, scaler, metrics


def train_decision_tree(df: pd.DataFrame, max_depth: int = 5):
    df_m = df[FEAT_DT + ['risco_incendio']].dropna()
    X, y = df_m[FEAT_DT], df_m['risco_incendio']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=50,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    metrics = {
        'report': classification_report(y_te, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_te, y_pred),
        'auc_roc': roc_auc_score(y_te, y_prob),
        'feature_importances': dict(zip(FEAT_DT, model.feature_importances_)),
        'rules': export_text(model, feature_names=FEAT_DT, max_depth=4),
    }
    return model, metrics


def save_knn(model, scaler, model_path: str, scaler_path: str) -> None:
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def save_dt(model, model_path: str) -> None:
    joblib.dump(model, model_path)
