"""Pipeline completo de treino e avaliação, sem o notebook.

Uso (a partir da raiz do repositório):
    python -m src.train
"""
import json

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, RECALL_ALVO, PASTA_RESULTADOS
from src.evaluate import metricas, pontos_de_operacao, metricas_por_subgrupo
from src.preprocessing import (carregar_dados, preparar_dados, separar_conjuntos,
                               criar_preprocessador)

ESPACO_XGB = {
    "n_estimators": [200, 300, 500],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
}


def criar_modelos(peso_positivos: float) -> dict:
    """Baseline e três modelos, todos compensando o desbalanceamento."""
    return {
        "baseline": Pipeline([("modelo", DummyClassifier(strategy="most_frequent"))]),
        "regressao_logistica": Pipeline([
            ("prep", criar_preprocessador()),
            ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "random_forest": Pipeline([
            ("modelo", RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20,
                                              class_weight="balanced", n_jobs=-1,
                                              random_state=RANDOM_STATE))]),
        "xgboost": Pipeline([
            ("modelo", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                                     scale_pos_weight=peso_positivos, eval_metric="logloss",
                                     n_jobs=-1, random_state=RANDOM_STATE))]),
    }


def buscar_hiperparametros(X_train, y_train, peso_positivos: float):
    """Busca aleatória com validação cruzada estratificada, otimizando o PR-AUC."""
    busca = RandomizedSearchCV(
        XGBClassifier(scale_pos_weight=peso_positivos, eval_metric="logloss",
                      n_jobs=-1, random_state=RANDOM_STATE),
        param_distributions=ESPACO_XGB, n_iter=15, scoring="average_precision",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
    )
    busca.fit(X_train, y_train)
    return busca


def main():
    df = preparar_dados(carregar_dados())
    X_train, X_val, X_test, y_train, y_val, y_test = separar_conjuntos(df)
    peso_positivos = (y_train == 0).sum() / (y_train == 1).sum()

    print("Comparando modelos na validação...")
    comparacao = {}
    for nome, modelo in criar_modelos(peso_positivos).items():
        modelo.fit(X_train, y_train)
        m = metricas(y_val, modelo.predict_proba(X_val)[:, 1])
        m.pop("matriz_confusao")
        comparacao[nome] = m
    print(pd.DataFrame(comparacao).T.round(3))

    print("\nBuscando hiperparâmetros do XGBoost...")
    busca = buscar_hiperparametros(X_train, y_train, peso_positivos)
    modelo_final = busca.best_estimator_
    print("Melhores parâmetros:", busca.best_params_)

    proba_val = modelo_final.predict_proba(X_val)[:, 1]
    pontos = pontos_de_operacao(y_val, proba_val)
    limiar = float(pontos.loc[pontos["criterio"] == f"Recall >= {RECALL_ALVO:.0%}", "limiar"].iloc[0])
    print("\nPontos de operação (validação):")
    print(pontos.round(3).to_string(index=False))

    proba_teste = modelo_final.predict_proba(X_test)[:, 1]
    teste = metricas(y_test, proba_teste, limiar)
    subgrupos = metricas_por_subgrupo(X_test, y_test, (proba_teste >= limiar).astype(int))
    print(f"\nTeste (limiar {limiar:.3f}):", {k: round(v, 4) for k, v in teste.items()
                                             if k != "matriz_confusao"})

    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo_final, "limiar": limiar}, PASTA_RESULTADOS / "modelo_final.joblib")
    resultados = {
        "comparacao_validacao": comparacao,
        "melhores_parametros": busca.best_params_,
        "limiar": limiar,
        "pontos_de_operacao": pontos.to_dict(orient="records"),
        "teste": teste,
        "subgrupos": subgrupos.to_dict(orient="records"),
    }
    with open(PASTA_RESULTADOS / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nResultados salvos em {PASTA_RESULTADOS}")


if __name__ == "__main__":
    main()
