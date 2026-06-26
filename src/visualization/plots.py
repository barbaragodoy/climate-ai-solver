"""Funções de visualização reutilizáveis para o projeto IC."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Matplotlib/Seaborn ──────────────────────────────────────────────────────

def plot_trend(df_anual: pd.DataFrame, col: str, title: str, ylabel: str, ax=None):
    show = ax is None
    if show:
        fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_anual.index, df_anual[col], color='steelblue')
    z = np.polyfit(range(len(df_anual)), df_anual[col], 1)
    ax.plot(df_anual.index, np.poly1d(z)(range(len(df_anual))), '--r',
            label=f'Tendência ({z[0]:+.4f}/ano)')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    if show:
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(cm: np.ndarray, labels: list, title: str, ax=None):
    show = ax is None
    if show:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')
    ax.set_title(title)
    if show:
        plt.tight_layout()
        plt.show()


# ── Plotly (para o Dashboard) ───────────────────────────────────────────────

def plotly_time_series(df: pd.DataFrame, col: str, title: str, color: str = '#1f77b4'):
    fig = px.line(df, x=df.index, y=col, title=title,
                  labels={col: col.replace('_', ' ').title(), 'index': 'Data'},
                  color_discrete_sequence=[color])
    fig.update_layout(hovermode='x unified', template='plotly_white')
    return fig


def plotly_monthly_climatology(df: pd.DataFrame):
    df_m = df.groupby('mes').agg(
        temp_max=('temp_max', 'mean'),
        temp_min=('temp_min', 'mean'),
        precipitacao=('precipitacao', 'mean'),
        umidade=('umidade_relativa', 'mean'),
    ).reset_index()

    meses = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
    df_m['mes_label'] = df_m['mes'].apply(lambda x: meses[x - 1])

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Bar(x=df_m['mes_label'], y=df_m['precipitacao'],
                         name='Precipitação (mm)', marker_color='steelblue', opacity=0.6),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df_m['mes_label'], y=df_m['temp_max'],
                             name='Temp. Máx', line=dict(color='tomato', width=2)),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=df_m['mes_label'], y=df_m['temp_min'],
                             name='Temp. Mín', line=dict(color='royalblue', width=2)),
                  secondary_y=True)
    fig.update_yaxes(title_text='Precipitação (mm)', secondary_y=False)
    fig.update_yaxes(title_text='Temperatura (°C)', secondary_y=True)
    fig.update_layout(title='Climatologia Mensal — Cerrado Goiano', template='plotly_white',
                      hovermode='x unified')
    return fig


def plotly_scatter_real_pred(y_real, y_pred, mae: float, r2: float, model_name: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_real, y=y_pred, mode='markers',
                             marker=dict(size=3, opacity=0.4, color='purple'),
                             name='Predições'))
    lim = [min(y_real) - 1, max(y_real) + 1]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode='lines',
                             line=dict(color='red', dash='dash'), name='Perfeito'))
    fig.update_layout(
        title=f'{model_name} — Real vs. Predito | MAE={mae:.2f}°C | R²={r2:.4f}',
        xaxis_title='Temperatura Real (°C)',
        yaxis_title='Temperatura Predita (°C)',
        template='plotly_white',
    )
    return fig
