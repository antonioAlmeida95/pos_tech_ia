"""Carga, limpeza, separação e pré-processamento dos dados."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (CAMINHO_CSV, RANDOM_STATE, BINARIAS, ORDINAIS, CONTINUAS,
                        FEATURES, ALVO_ORIGINAL, ALVO, IMC_MAXIMO)


def carregar_dados(caminho=CAMINHO_CSV) -> pd.DataFrame:
    """Lê o CSV e converte cada grupo de variáveis para o tipo inteiro adequado."""
    df = pd.read_csv(caminho)
    assert set(FEATURES) == set(df.columns) - {ALVO_ORIGINAL}, "Colunas inesperadas no CSV"
    df[BINARIAS + ORDINAIS + [ALVO_ORIGINAL]] = df[BINARIAS + ORDINAIS + [ALVO_ORIGINAL]].astype("int8")
    df[CONTINUAS] = df[CONTINUAS].astype("int16")
    return df


def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Cria o alvo binário, remove duplicatas e limita o IMC.

    Pré-diabetes (classe 1) é agrupada com os não diabéticos.
    """
    df = df.copy()
    df[ALVO] = (df[ALVO_ORIGINAL] == 2).astype("int8")
    df = df.drop_duplicates().reset_index(drop=True)
    df["BMI"] = df["BMI"].clip(upper=IMC_MAXIMO)
    return df


def separar_conjuntos(df: pd.DataFrame):
    """Divide em treino (70%), validação (15%) e teste (15%), estratificado pelo alvo."""
    X, y = df[FEATURES], df[ALVO]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test


def criar_preprocessador() -> ColumnTransformer:
    """Padroniza contínuas e ordinais; mantém as binárias como estão."""
    return ColumnTransformer([
        ("escala", StandardScaler(), CONTINUAS + ORDINAIS),
        ("binarias", "passthrough", BINARIAS),
    ])
