"""
Pré-processamento de dados climáticos: limpeza, imputação, normalização e
engenharia de features para uso nos modelos.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path


NUMERIC_COLS = [
    "temp_max", "temp_min", "temp_media",
    "precipitacao", "umidade_relativa",
    "velocidade_vento", "radiacao_solar",
]


def load_raw(path: str = "data/raw/cerrado_clima_raw.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col="data", parse_dates=True)
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche lacunas com interpolação linear + ffill/bfill nas bordas."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].interpolate(method="time").ffill().bfill()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features derivadas para os modelos."""
    df = df.copy()
    df["amplitude_termica"] = df["temp_max"] - df["temp_min"]
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["dia_sin"] = np.sin(2 * np.pi * df["dia_do_ano"] / 365)
    df["dia_cos"] = np.cos(2 * np.pi * df["dia_do_ano"] / 365)
    df["precip_7d"] = df["precipitacao"].rolling(7, min_periods=1).sum()
    df["temp_media_30d"] = df["temp_media"].rolling(30, min_periods=1).mean()
    df["estacao_seca"] = df["mes"].isin([4, 5, 6, 7, 8, 9]).astype(int)
    return df


def scale_features(df: pd.DataFrame, cols: list, method: str = "minmax"):
    """
    Normaliza colunas numéricas.
    Retorna (df_scaled, scaler).
    """
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    df_scaled = df.copy()
    df_scaled[cols] = scaler.fit_transform(df[cols])
    return df_scaled, scaler


def make_sequences(series: np.ndarray, n_steps: int = 30):
    """
    Converte série 1-D em pares (X, y) para modelos LSTM.
    X.shape = (n_samples, n_steps, 1)
    y.shape = (n_samples,)
    """
    X, y = [], []
    for i in range(n_steps, len(series)):
        X.append(series[i - n_steps:i])
        y.append(series[i])
    X = np.array(X).reshape(-1, n_steps, 1)
    y = np.array(y)
    return X, y


def get_processed_data(raw_path: str = "data/raw/cerrado_clima_raw.csv") -> pd.DataFrame:
    """Pipeline completo: carrega → limpa → adiciona features."""
    df = load_raw(raw_path)
    df = fill_missing(df)
    df = add_features(df)
    return df


def save_processed(df: pd.DataFrame, path: str = "data/processed/cerrado_clima_processado.csv") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"Dados processados salvos: {path}")
