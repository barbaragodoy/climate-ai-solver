"""
Gerador de dados climáticos sintéticos para o Cerrado Goiano (Goiânia, GO).
Reproduz padrões sazonais reais: estação chuvosa (out-mar) e seca (abr-set).
Fonte de referência: INMET, estação A002 - Goiânia.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)

# Parâmetros mensais médios históricos do Cerrado (Goiânia)
# [temperatura_media, precipitacao_media_mm, umidade_relativa_%, vel_vento_m/s, rad_solar_MJ/m2]
MONTHLY_PARAMS = {
    1:  {"temp_mean": 24.5, "temp_std": 2.0, "precip_mean": 230, "humidity": 82, "wind": 2.1, "radiation": 18.5},
    2:  {"temp_mean": 24.7, "temp_std": 2.0, "precip_mean": 200, "humidity": 80, "wind": 2.0, "radiation": 18.8},
    3:  {"temp_mean": 24.5, "temp_std": 2.0, "precip_mean": 190, "humidity": 80, "wind": 1.9, "radiation": 18.2},
    4:  {"temp_mean": 24.0, "temp_std": 2.2, "precip_mean": 100, "humidity": 72, "wind": 1.8, "radiation": 17.5},
    5:  {"temp_mean": 22.5, "temp_std": 2.5, "precip_mean": 30,  "humidity": 62, "wind": 1.7, "radiation": 16.8},
    6:  {"temp_mean": 21.0, "temp_std": 2.8, "precip_mean": 8,   "humidity": 52, "wind": 1.8, "radiation": 16.2},
    7:  {"temp_mean": 20.5, "temp_std": 3.0, "precip_mean": 5,   "humidity": 48, "wind": 2.0, "radiation": 16.5},
    8:  {"temp_mean": 22.5, "temp_std": 3.0, "precip_mean": 10,  "humidity": 38, "wind": 2.2, "radiation": 18.0},
    9:  {"temp_mean": 24.5, "temp_std": 3.0, "precip_mean": 45,  "humidity": 45, "wind": 2.3, "radiation": 19.2},
    10: {"temp_mean": 25.0, "temp_std": 2.5, "precip_mean": 120, "humidity": 68, "wind": 2.2, "radiation": 19.5},
    11: {"temp_mean": 24.5, "temp_std": 2.2, "precip_mean": 200, "humidity": 78, "wind": 2.0, "radiation": 18.8},
    12: {"temp_mean": 24.2, "temp_std": 2.0, "precip_mean": 230, "humidity": 82, "wind": 2.0, "radiation": 18.5},
}

# Taxa de aquecimento anual (tendência climática: +0.25°C/década)
WARMING_RATE_PER_YEAR = 0.025


def _trend(year: int, base_year: int = 1974) -> float:
    return (year - base_year) * WARMING_RATE_PER_YEAR


def generate_daily_climate(start_year: int = 1974, end_year: int = 2024) -> pd.DataFrame:
    """
    Gera série temporal diária de dados climáticos para o Cerrado Goiano.

    Variáveis geradas:
        - temp_max, temp_min, temp_media (°C)
        - precipitacao (mm)
        - umidade_relativa (%)
        - velocidade_vento (m/s)
        - radiacao_solar (MJ/m²)

    Labels derivados:
        - risco_onda_calor (1 = dia de risco)
        - risco_incendio  (1 = risco elevado)
    """
    records = []

    date_range = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        freq="D"
    )

    for date in date_range:
        m = date.month
        y = date.year
        p = MONTHLY_PARAMS[m]
        trend = _trend(y)

        # Temperatura com sazonalidade + tendência + ruído
        temp_med = p["temp_mean"] + trend + np.random.normal(0, p["temp_std"])
        amplitude = np.random.uniform(7, 12)
        temp_max = temp_med + amplitude / 2
        temp_min = temp_med - amplitude / 2

        # Precipitação: distribuição gamma (assimétrica como precipitação real)
        if p["precip_mean"] > 0:
            shape = 0.6
            scale = p["precip_mean"] / shape
            precip_daily_mean = p["precip_mean"] / 30
            if precip_daily_mean > 0:
                precip = np.random.gamma(shape, precip_daily_mean / shape) if np.random.random() < 0.45 else 0.0
            else:
                precip = 0.0
        else:
            precip = 0.0
        precip = max(0.0, round(precip, 1))

        # Umidade relativa (correlacionada com precipitação)
        humidity_base = p["humidity"] + trend * (-0.1)
        humidity = np.clip(humidity_base + np.random.normal(0, 8) + (precip > 0) * 10, 10, 100)

        # Velocidade do vento
        wind = max(0.1, p["wind"] + np.random.normal(0, 0.4))

        # Radiação solar (inversamente correlacionada com nuvens/precipitação)
        radiation = max(5, p["radiation"] - (precip > 0) * 3 + np.random.normal(0, 1.5))

        # --- Labels ---
        # Onda de calor: temp_max >= 35°C E umidade < 40% (critério climatológico adaptado ao Cerrado)
        risco_onda_calor = int(temp_max >= 35.0 and humidity < 40.0)

        # Risco de incêndio: estação seca + temperatura alta + umidade baixa + vento forte
        fwi_proxy = (temp_max - 25) * 0.3 + (100 - humidity) * 0.04 + wind * 0.5 - precip * 0.1
        risco_incendio = int(fwi_proxy > 5.0 and m in [6, 7, 8, 9] and humidity < 35)

        records.append({
            "data": date,
            "ano": y,
            "mes": m,
            "dia_do_ano": date.dayofyear,
            "temp_max": round(temp_max, 1),
            "temp_min": round(temp_min, 1),
            "temp_media": round(temp_med, 1),
            "precipitacao": precip,
            "umidade_relativa": round(humidity, 1),
            "velocidade_vento": round(wind, 2),
            "radiacao_solar": round(radiation, 2),
            "risco_onda_calor": risco_onda_calor,
            "risco_incendio": risco_incendio,
        })

    df = pd.DataFrame(records)
    df.set_index("data", inplace=True)
    return df


def inject_missing_values(df: pd.DataFrame, missing_rate: float = 0.02) -> pd.DataFrame:
    """Injeta valores faltantes aleatórios para simular dados reais do INMET."""
    df = df.copy()
    numeric_cols = ["temp_max", "temp_min", "temp_media", "precipitacao",
                    "umidade_relativa", "velocidade_vento", "radiacao_solar"]
    rng = np.random.default_rng(SEED)
    for col in numeric_cols:
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan
    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"Dataset salvo: {path} ({len(df):,} registros)")


if __name__ == "__main__":
    print("Gerando dados climáticos do Cerrado Goiano (1974–2024)...")
    df_raw = generate_daily_climate(1974, 2024)
    df_with_missing = inject_missing_values(df_raw, missing_rate=0.02)

    save_dataset(df_with_missing, "data/raw/cerrado_clima_raw.csv")
    save_dataset(df_raw, "data/raw/cerrado_clima_completo.csv")

    print("\nEstatísticas básicas:")
    print(df_raw[["temp_max", "temp_min", "precipitacao", "umidade_relativa"]].describe().round(2))
    print(f"\nDias com risco de onda de calor: {df_raw['risco_onda_calor'].sum():,}")
    print(f"Dias com risco de incêndio:      {df_raw['risco_incendio'].sum():,}")
