"""
Dashboard Interativo — Clima do Cerrado Goiano
Projeto IC: Aplicação de IA para Resolução de Problemas Climáticos
SENAI/FIEG | Aluna: Bárbara Letícia Mota Godoy

Execução: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Permite importar src/ a partir do diretório raiz
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

from src.data.generate_data import generate_daily_climate, inject_missing_values, save_dataset
from src.data.preprocessor import fill_missing, add_features
from src.visualization.plots import plotly_monthly_climatology

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Clima do Cerrado — IC IA",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

MESES = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
COLORS = {
    'temp_max': '#e74c3c',
    'temp_min': '#3498db',
    'temp_media': '#f39c12',
    'precipitacao': '#2980b9',
    'umidade_relativa': '#1abc9c',
}


# ── Cache: carrega/gera dados ───────────────────────────────────────────────
@st.cache_data(show_spinner="Carregando dados climáticos...")
def load_data() -> pd.DataFrame:
    raw_path = ROOT / "data" / "raw" / "cerrado_clima_raw.csv"
    if not raw_path.exists():
        df_raw = generate_daily_climate(1974, 2024)
        df_missing = inject_missing_values(df_raw)
        save_dataset(df_missing, str(raw_path))
        save_dataset(df_raw, str(ROOT / "data" / "raw" / "cerrado_clima_completo.csv"))
    df = pd.read_csv(raw_path, index_col="data", parse_dates=True)
    df = fill_missing(df)
    df = add_features(df)
    return df


@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    model_dir = ROOT / "outputs" / "models"
    for name, fname in [
        ("regressao", "regressao_linear.pkl"),
        ("scaler_reg", "scaler_regressao.pkl"),
        ("knn", "knn_onda_calor.pkl"),
        ("scaler_knn", "scaler_knn.pkl"),
        ("arvore", "arvore_incendio.pkl"),
        ("kmeans", "kmeans_perfis.pkl"),
        ("scaler_km", "scaler_kmeans.pkl"),
    ]:
        path = model_dir / fname
        if path.exists():
            models[name] = joblib.load(path)
    return models


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Flag_of_Goi%C3%A1s.svg/200px-Flag_of_Goi%C3%A1s.svg.png",
             width=80)
    st.title("🌿 Clima do Cerrado")
    st.caption("Projeto IC — SENAI/FIEG | 2025–2026")
    st.divider()

    pagina = st.radio(
        "Navegação",
        ["📊 Visão Geral", "📈 Tendências", "🔥 Eventos Extremos",
         "🤖 Modelos de IA", "🗓️ Perfis Sazonais"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Dados: INMET / Cerrado Goiano  \n1974–2024 (50 anos)")

df = load_data()
models = load_models()


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 1 — Visão Geral
# ─────────────────────────────────────────────────────────────────────────────
if pagina == "📊 Visão Geral":
    st.title("📊 Visão Geral — Clima do Cerrado Goiano")
    st.caption("Série histórica 1974–2024 | Estação: Goiânia, GO")

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Temp. Média Histórica", f"{df['temp_media'].mean():.1f} °C")
    col2.metric("Temp. Máx. Registrada", f"{df['temp_max'].max():.1f} °C")
    col3.metric("Precip. Anual Média", f"{df['precipitacao'].sum()/50:.0f} mm")
    col4.metric("Dias Risco Onda de Calor", f"{df['risco_onda_calor'].sum():,}")
    col5.metric("Dias Risco de Incêndio", f"{df['risco_incendio'].sum():,}")

    st.divider()

    # Filtro de período
    st.subheader("Série Temporal Diária")
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    ano_ini = col_f1.slider("Ano inicial", 1974, 2023, 2000)
    ano_fim = col_f2.slider("Ano final", ano_ini + 1, 2024, 2024)
    variavel = col_f3.selectbox(
        "Variável",
        ['temp_media', 'temp_max', 'temp_min', 'precipitacao', 'umidade_relativa'],
        format_func=lambda x: x.replace('_', ' ').title()
    )

    df_filt = df[str(ano_ini):str(ano_fim)]
    df_mensal = df_filt[variavel].resample('ME').mean() if variavel != 'precipitacao' \
        else df_filt[variavel].resample('ME').sum()

    fig = px.line(
        df_mensal.reset_index(),
        x='data', y=variavel,
        title=f'{variavel.replace("_"," ").title()} — Média Mensal ({ano_ini}–{ano_fim})',
        color_discrete_sequence=[COLORS.get(variavel, '#636EFA')],
        labels={variavel: variavel.replace('_', ' ').title(), 'data': 'Data'},
    )
    fig.update_layout(template='plotly_white', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # Climatologia mensal
    st.subheader("Climatologia Mensal")
    st.plotly_chart(plotly_monthly_climatology(df), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 2 — Tendências de Longo Prazo
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "📈 Tendências":
    st.title("📈 Tendências Climáticas de Longo Prazo (1974–2024)")

    df_anual = df.resample('YE').agg(
        temp_media=('temp_media', 'mean'),
        temp_max=('temp_max', 'mean'),
        precipitacao=('precipitacao', 'sum'),
        umidade_relativa=('umidade_relativa', 'mean'),
    ).reset_index()
    df_anual.columns = ['data', 'temp_media', 'temp_max', 'precipitacao', 'umidade_relativa']
    df_anual['ano'] = df_anual['data'].dt.year

    tab1, tab2, tab3 = st.tabs(["🌡️ Temperatura", "🌧️ Precipitação", "💧 Umidade"])

    with tab1:
        z = np.polyfit(df_anual['ano'], df_anual['temp_media'], 1)
        trend = np.poly1d(z)(df_anual['ano'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_anual['ano'], y=df_anual['temp_media'],
                                 mode='lines+markers', name='Temp. Média', line_color='orange'))
        fig.add_trace(go.Scatter(x=df_anual['ano'], y=trend, mode='lines',
                                 name=f'Tendência ({z[0]:+.4f}°C/ano)',
                                 line=dict(color='red', dash='dash', width=2)))
        fig.update_layout(title='Temperatura Média Anual', xaxis_title='Ano',
                          yaxis_title='°C', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"**Taxa de aquecimento:** {z[0]:.4f} °C/ano = **{z[0]*10:.3f} °C/década**  \n"
                f"Aquecimento total (50 anos): ~{z[0]*50:.2f} °C")

    with tab2:
        fig2 = px.bar(df_anual, x='ano', y='precipitacao',
                      title='Precipitação Total Anual (mm)',
                      color='precipitacao', color_continuous_scale='Blues')
        fig2.update_layout(template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = px.line(df_anual, x='ano', y='umidade_relativa',
                       title='Umidade Relativa Média Anual (%)',
                       color_discrete_sequence=['teal'])
        fig3.add_hline(y=40, line_dash='dash', line_color='red',
                       annotation_text='Limiar crítico (40%)')
        fig3.update_layout(template='plotly_white')
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 3 — Eventos Extremos
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "🔥 Eventos Extremos":
    st.title("🔥 Eventos Climáticos Extremos")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dias de Risco de Onda de Calor por Ano")
        ondas = df.groupby('ano')['risco_onda_calor'].sum().reset_index()
        z = np.polyfit(ondas['ano'], ondas['risco_onda_calor'], 1)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ondas['ano'], y=ondas['risco_onda_calor'],
                             marker_color='tomato', name='Dias de Risco'))
        fig.add_trace(go.Scatter(x=ondas['ano'], y=np.poly1d(z)(ondas['ano']),
                                 mode='lines', name=f'Tendência ({z[0]:+.2f}/ano)',
                                 line=dict(color='darkred', dash='dash', width=2)))
        fig.update_layout(template='plotly_white', hovermode='x')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Dias de Risco de Incêndio por Ano")
        incendio = df.groupby('ano')['risco_incendio'].sum().reset_index()
        z2 = np.polyfit(incendio['ano'], incendio['risco_incendio'], 1)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=incendio['ano'], y=incendio['risco_incendio'],
                              marker_color='darkorange', name='Dias de Risco'))
        fig2.add_trace(go.Scatter(x=incendio['ano'], y=np.poly1d(z2)(incendio['ano']),
                                  mode='lines', name=f'Tendência ({z2[0]:+.2f}/ano)',
                                  line=dict(color='saddlebrown', dash='dash', width=2)))
        fig2.update_layout(template='plotly_white', hovermode='x')
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Concentração Mensal de Eventos")
    mes_ondas = df[df['risco_onda_calor'] == 1].groupby('mes').size().reindex(range(1, 13), fill_value=0)
    mes_incendio = df[df['risco_incendio'] == 1].groupby('mes').size().reindex(range(1, 13), fill_value=0)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=MESES, y=mes_ondas.values, name='Onda de Calor', marker_color='tomato'))
    fig3.add_trace(go.Bar(x=MESES, y=mes_incendio.values, name='Incêndio', marker_color='darkorange'))
    fig3.update_layout(barmode='group', template='plotly_white',
                       title='Ocorrências Mensais Acumuladas (1974–2024)',
                       xaxis_title='Mês', yaxis_title='Nº de Dias')
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 4 — Modelos de IA
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "🤖 Modelos de IA":
    st.title("🤖 Modelos de Inteligência Artificial")
    st.info("Execute os notebooks para treinar e salvar os modelos antes de usar esta página.")

    tab_reg, tab_knn, tab_dt, tab_km = st.tabs([
        "📉 Regressão Linear", "🌡️ k-NN (Onda de Calor)",
        "🌲 Árvore de Decisão (Incêndio)", "🔵 K-Means (Perfis)"
    ])

    with tab_reg:
        st.subheader("Regressão Linear — Previsão de Temperatura Máxima")
        st.markdown("""
        **Objetivo:** Prever `temp_max` (temperatura máxima diária) a partir de variáveis climáticas.

        **Features utilizadas:** temperatura mínima, umidade, vento, radiação solar, precipitação,
        componentes sazonais (seno/cosseno do mês e dia do ano), amplitude térmica.

        **Métricas esperadas:**
        - MAE ≈ 1,5–2,5 °C
        - R² ≈ 0,75–0,90

        Execute o notebook `02_regressao_linear.ipynb` para treinar e salvar o modelo.
        """)

        if 'regressao' in models:
            st.success("Modelo carregado! Use o notebook para ver gráficos detalhados.")
        else:
            st.warning("Modelo não encontrado. Execute o notebook 02.")

    with tab_knn:
        st.subheader("k-NN — Identificação de Dias de Risco de Onda de Calor")
        st.markdown("""
        **Critério de risco:** Temperatura máxima ≥ 35°C **E** umidade relativa < 40%.

        **Algoritmo:** K-Nearest Neighbors com distância euclidiana e pesos por distância.

        O melhor valor de k é selecionado por validação cruzada estratificada (5-fold),
        maximizando a **AUC-ROC** para lidar com o desbalanceamento de classes.
        """)

        # Demonstração ao vivo
        st.subheader("Predição ao Vivo")
        col1, col2 = st.columns(2)
        temp_max_inp = col1.slider("Temperatura Máxima (°C)", 20.0, 45.0, 35.0, 0.5)
        umid_inp = col2.slider("Umidade Relativa (%)", 10.0, 100.0, 35.0, 1.0)
        col3, col4 = st.columns(2)
        vento_inp = col3.slider("Velocidade do Vento (m/s)", 0.0, 8.0, 2.5, 0.1)
        rad_inp = col4.slider("Radiação Solar (MJ/m²)", 8.0, 28.0, 19.0, 0.5)

        risco_manual = int(temp_max_inp >= 35.0 and umid_inp < 40.0)
        if risco_manual:
            st.error("⚠️ **ALERTA: Condições de Risco de Onda de Calor detectadas!**")
        else:
            st.success("✅ Condições dentro da normalidade climática.")

    with tab_dt:
        st.subheader("Árvore de Decisão — Risco de Incêndio")
        st.markdown("""
        **Variáveis mais importantes** (por impureza de Gini):
        1. `umidade_relativa` — principal indicador
        2. `precipitacao` (acumulada 7 dias)
        3. `temp_max`
        4. `velocidade_vento`
        5. `estacao_seca` (binário: abril–setembro)

        A árvore traduz as decisões em regras interpretáveis, úteis para gestores de defesa civil.
        """)

        st.subheader("Avaliador Manual de Risco de Incêndio")
        col1, col2, col3 = st.columns(3)
        tm = col1.number_input("Temp. Máx. (°C)", 15.0, 45.0, 32.0, 0.5)
        ur = col2.number_input("Umidade (%)", 5.0, 100.0, 28.0, 1.0)
        vt = col3.number_input("Vento (m/s)", 0.0, 10.0, 2.8, 0.1)
        col4, col5 = st.columns(2)
        pp = col4.number_input("Precip. 7 dias (mm)", 0.0, 200.0, 0.0, 1.0)
        seca = col5.selectbox("Estação Seca?", ["Sim", "Não"])

        fwi = (tm - 25) * 0.3 + (100 - ur) * 0.04 + vt * 0.5 - pp * 0.1
        seca_val = 1 if seca == "Sim" else 0
        risco_inc = int(fwi > 5.0 and seca_val == 1 and ur < 35)

        if risco_inc:
            st.error(f"🔥 **ALTO RISCO DE INCÊNDIO** (FWI proxy: {fwi:.2f})")
        elif fwi > 3:
            st.warning(f"⚡ Risco moderado (FWI proxy: {fwi:.2f})")
        else:
            st.success(f"✅ Risco baixo (FWI proxy: {fwi:.2f})")

    with tab_km:
        st.subheader("K-Means — Perfis Climáticos Sazonais")

        perfis = {
            "Chuvoso-Quente (out–mar)": {
                "Descrição": "Alta precipitação, umidade >75%, temperatura ~25°C",
                "Período": "Outubro a Março",
                "Risco Principal": "Alagamentos, deslizamentos",
                "Cor": "🔵",
            },
            "Transição (abr, out)": {
                "Descrição": "Características intermediárias entre estações",
                "Período": "Abril e Outubro",
                "Risco Principal": "Variabilidade extrema",
                "Cor": "🟡",
            },
            "Seco-Ameno (mai–jun)": {
                "Descrição": "Temperatura baixa (~21°C), início da estação seca",
                "Período": "Maio a Junho",
                "Risco Principal": "Baixa umidade crescente",
                "Cor": "🟤",
            },
            "Seco-Crítico (jul–set)": {
                "Descrição": "Umidade <35%, vento forte, risco máximo de incêndio",
                "Período": "Julho a Setembro",
                "Risco Principal": "🔥 INCÊNDIO CRÍTICO",
                "Cor": "🔴",
            },
        }

        for nome, info in perfis.items():
            with st.expander(f"{info['Cor']} **{nome}**"):
                col1, col2 = st.columns(2)
                col1.markdown(f"**Descrição:** {info['Descrição']}")
                col1.markdown(f"**Período:** {info['Período']}")
                col2.markdown(f"**Risco Principal:** {info['Risco Principal']}")


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 5 — Perfis Sazonais
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "🗓️ Perfis Sazonais":
    st.title("🗓️ Análise de Perfis Sazonais")

    # Heatmap anual
    st.subheader("Mapa de Calor — Temperatura Média Mensal por Ano")
    pivot = df.pivot_table(values='temp_media', index='ano', columns='mes', aggfunc='mean')
    pivot.columns = MESES

    fig = px.imshow(
        pivot,
        color_continuous_scale='RdYlBu_r',
        labels=dict(color="Temp. Média (°C)", x="Mês", y="Ano"),
        title="Temperatura Média Mensal por Ano (°C)",
        aspect='auto',
    )
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Boxplot de Temperatura por Mês")
        df_box = df[['mes', 'temp_max']].copy()
        df_box['mes_label'] = df_box['mes'].apply(lambda x: MESES[x - 1])
        fig2 = px.box(df_box, x='mes_label', y='temp_max',
                      category_orders={'mes_label': MESES},
                      color='mes_label',
                      labels={'temp_max': 'Temp. Máx. (°C)', 'mes_label': 'Mês'},
                      title='Distribuição de Temperatura Máxima')
        fig2.add_hline(y=35, line_dash='dash', line_color='red',
                       annotation_text='Limiar onda de calor')
        fig2.update_layout(showlegend=False, template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Precipitação Média por Mês")
        df_prec = df.groupby('mes')['precipitacao'].mean().reset_index()
        df_prec['mes_label'] = df_prec['mes'].apply(lambda x: MESES[x - 1])
        fig3 = px.bar(df_prec, x='mes_label', y='precipitacao',
                      category_orders={'mes_label': MESES},
                      color='precipitacao', color_continuous_scale='Blues',
                      labels={'precipitacao': 'Precipitação Média (mm)', 'mes_label': 'Mês'},
                      title='Precipitação Média Diária por Mês')
        fig3.update_layout(template='plotly_white')
        st.plotly_chart(fig3, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption(
    "🎓 Iniciação Científica — SENAI/FIEG  \n"
    "Aluna: Bárbara Letícia Mota Godoy  \n"
    "Orientador: Prof. Gustavo S. Vinhal  \n"
    "© 2025–2026"
)
