"""Módulo de regressão linear para previsão de temperatura."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


FEATURES = [
    'temp_min', 'umidade_relativa', 'velocidade_vento',
    'radiacao_solar', 'precipitacao', 'mes_sin', 'mes_cos',
    'dia_sin', 'dia_cos', 'amplitude_termica', 'precip_7d', 'temp_media_30d'
]
TARGET = 'temp_max'


def train(df: pd.DataFrame, alpha: float = 0.0):
    df_m = df[FEATURES + [TARGET]].dropna()
    X, y = df_m[FEATURES], df_m[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = Ridge(alpha=alpha) if alpha > 0 else LinearRegression()
    model.fit(X_tr_s, y_tr)

    y_pred = model.predict(X_te_s)
    metrics = {
        'MAE':  mean_absolute_error(y_te, y_pred),
        'MSE':  mean_squared_error(y_te, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_te, y_pred)),
        'R2':   r2_score(y_te, y_pred),
    }
    return model, scaler, metrics


def predict(model, scaler, input_df: pd.DataFrame) -> np.ndarray:
    X = scaler.transform(input_df[FEATURES])
    return model.predict(X)


def save(model, scaler, model_path: str, scaler_path: str) -> None:
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def load(model_path: str, scaler_path: str):
    return joblib.load(model_path), joblib.load(scaler_path)
