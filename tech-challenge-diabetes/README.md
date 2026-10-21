# Diagnóstico de Diabetes com Machine Learning — Tech Challenge Fase 1

Modelo de triagem que identifica pacientes com provável diabetes a partir de 21 indicadores de saúde, para apoiar o médico na decisão de solicitar exames de confirmação. O modelo não substitui o diagnóstico: o médico tem sempre a palavra final.

## Dataset

**CDC BRFSS 2015 — Diabetes Health Indicators** (253.680 respostas de uma pesquisa telefônica de saúde nos EUA, 21 indicadores).

- Download: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- Arquivo usado: `diabetes_012_health_indicators_BRFSS2015.csv`, já incluído em `data/`

O grupo avaliou inicialmente um dataset de câncer de próstata, descartado por não conter sinal aprendível (ROC-AUC ≈ 0,5).

## Como executar

### Com Docker

```bash
docker build -t diabetes-ml .
docker run -p 8888:8888 diabetes-ml
```

Abra http://localhost:8888 e execute `notebooks/diabetes_diagnosis_final.ipynb` (Kernel → Restart & Run All).

### Sem Docker

```bash
pip install -r requirements.txt
jupyter lab                 # notebook com a análise completa
python -m src.train         # ou: pipeline completo pelo terminal
```

O script `src/train.py` treina e avalia o modelo sem o notebook e grava `reports/resultados.json` e `reports/modelo_final.joblib`. A execução completa leva cerca de 2 minutos.

### Google Colab
As constantes do projeto (caminhos, grupos de variáveis, semente aleatória) ficam em `src/config.py`, compartilhadas pelo notebook e pelo script `src/train.py`. Os caminhos são calculados a partir da raiz do repositório, então o notebook funciona tanto localmente quanto no contêiner Docker.

São necessários zipar os arquivos:

1. `src/__init__.py`
2. `src/config.py`
3. `data/diabetes_012_health_indicators_BRFSS2015.csv`

## Abordagem

1. **Exploração:** estatísticas descritivas, distribuições por classe e correlação de Spearman.
2. **Pré-processamento:** alvo binário (diabetes contra não diabetes/pré-diabetes), remoção de duplicatas, limitação do IMC em 60, pipeline com `ColumnTransformer`.
3. **Modelagem:** baseline, Regressão Logística, Random Forest e XGBoost, com peso nas classes para compensar o desbalanceamento (15% de positivos); busca de hiperparâmetros com validação cruzada.
4. **Limiar:** escolhido na validação para atingir recall de 90%.
5. **Interpretação:** importância de features, SHAP (global e por paciente) e odds ratios.
6. **Análise crítica:** desempenho por sexo e renda, limitações e uso prático.

## Resultados (conjunto de teste, 34.468 pacientes)

| Métrica | Valor |
|---|---|
| Recall | 0,898 |
| Precisão | 0,265 |
| F1 | 0,409 |
| ROC-AUC | 0,817 |
| PR-AUC | 0,435 |

O modelo detectou 4.726 dos 5.265 diabéticos do teste. A precisão baixa é uma escolha deliberada: em triagem, um alarme falso se resolve com um exame de glicemia, enquanto um diabético não detectado fica sem tratamento.

## Estrutura do repositório

```
├── data/                  # datasets
├── notebooks/
│   └── diabetes_diagnosis_final.ipynb
├── src/
│   ├── config.py          # caminhos, colunas e constantes
│   ├── preprocessing.py   # carga, limpeza, split e pré-processador
│   ├── evaluate.py        # métricas, pontos de operação, subgrupos
│   └── train.py           # pipeline completo pelo terminal
├── reports/
│   └── figures/           # gráficos gerados pelo notebook
├── requirements.txt
├── Dockerfile
└── README.md
```

## Equipe

| Nome | RM |
|---|---|
|Antonio Lucas de Almeida|rm378137|
|Arthur Gradim Givigir|rm378714|
|Fabrícia de Jesus Santos|rm378001|
|Pedro Couto Araújo Júnior|rm378221|
