# Climate AI Solver

Projeto de iniciação científica voltado ao estudo de dados climáticos do Cerrado Goiano e à aplicação de técnicas de Inteligência Artificial para análise, previsão e identificação de padrões extremos.

## Objetivo

Este repositório reúne:
- uma pipeline de geração e processamento de dados climáticos;
- notebooks com análise exploratória e experimentos de modelos;
- uma aplicação interativa em Streamlit para visualização dos dados;
- modelos de regressão, classificação, clustering e previsão temporal.

## Estrutura do projeto

- `dashboard/` — aplicação web interativa em Streamlit.
- `data/` — dados brutos e processados.
- `docs/` — documentos e referências do projeto.
- `notebooks/` — notebooks de exploração e experimentos.
- `outputs/` — figuras e modelos treinados.
- `src/` — módulos de geração de dados, pré-processamento, visualização e modelos.

## Requisitos

O projeto foi desenvolvido com Python 3.10+.

Instale as dependências com:

```bash
pip install -r requirements.txt
```

## Como executar

### Dashboard

```bash
streamlit run dashboard/app.py
```

### Notebooks

Os notebooks podem ser abertos com Jupyter:

```bash
jupyter notebook
```

## Fluxo principal

1. Geração de dados climáticos sintéticos ou reais.
2. Limpeza e enriquecimento das variáveis.
3. Exploração dos dados em notebooks.
4. Treinamento e avaliação de modelos de IA.
5. Visualização interativa no dashboard.

## Licença

Este projeto é de uso acadêmico e pode ser adaptado conforme necessidade do projeto de pesquisa.
