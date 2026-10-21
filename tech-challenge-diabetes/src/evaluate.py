"""Métricas, escolha do limiar e análise por subgrupo."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, precision_recall_curve,
                             confusion_matrix)


def metricas(y_true, proba, limiar: float = 0.5) -> dict:
    """Métricas de classificação para um limiar de decisão."""
    pred = (proba >= limiar).astype(int)
    return {
        "acuracia": accuracy_score(y_true, pred),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precisao": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "matriz_confusao": confusion_matrix(y_true, pred).tolist(),
    }


def pontos_de_operacao(y_true, proba, recalls=(0.80, 0.85, 0.90)) -> pd.DataFrame:
    """Para cada recall-alvo, o maior limiar que o atinge, mais o ponto de melhor F1.

    Inclui a tradução em pacientes a cada 1.000 atendidos.
    """
    precisao, recall, limiares = precision_recall_curve(y_true, proba)
    precisao, recall = precisao[:-1], recall[:-1]

    opcoes = []
    for alvo in recalls:
        i = np.where(recall >= alvo)[0][-1]
        opcoes.append((f"Recall >= {alvo:.0%}", limiares[i], recall[i], precisao[i]))
    f1 = 2 * precisao * recall / (precisao + recall + 1e-12)
    i = f1.argmax()
    opcoes.append(("Melhor F1", limiares[i], recall[i], precisao[i]))

    pontos = pd.DataFrame(opcoes, columns=["criterio", "limiar", "recall", "precisao"])
    prevalencia = np.mean(y_true)
    pontos["detectados_por_1000"] = (pontos["recall"] * prevalencia * 1000).round()
    pontos["alarmes_falsos_por_1000"] = (pontos["detectados_por_1000"] * (1 / pontos["precisao"] - 1)).round()
    pontos["perdidos_por_1000"] = (prevalencia * 1000 - pontos["detectados_por_1000"]).round()
    return pontos


def metricas_por_subgrupo(X: pd.DataFrame, y_true, pred) -> pd.DataFrame:
    """Recall e precisão por sexo e por faixa de renda."""
    resultado = X.assign(real=np.asarray(y_true), previsto=np.asarray(pred))
    resultado["Sexo"] = resultado["Sex"].map({0: "Feminino", 1: "Masculino"})
    resultado["Faixa de renda"] = pd.cut(resultado["Income"], bins=[0, 3, 6, 8],
                                         labels=["Baixa (1-3)", "Media (4-6)", "Alta (7-8)"])
    linhas = []
    for coluna in ["Sexo", "Faixa de renda"]:
        for grupo, dados in resultado.groupby(coluna, observed=True):
            linhas.append({
                "variavel": coluna, "grupo": str(grupo), "pacientes": len(dados),
                "prevalencia": dados["real"].mean(),
                "recall": recall_score(dados["real"], dados["previsto"]),
                "precisao": precision_score(dados["real"], dados["previsto"]),
            })
    return pd.DataFrame(linhas)
