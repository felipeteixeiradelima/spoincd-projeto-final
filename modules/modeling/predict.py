"""
Módulo de inferência e predição operacional do nível de risco criminal.
"""

from pathlib import Path
from typing import Dict, Union

import joblib
import numpy as np
import pandas as pd

from modules.config import FINAL_MODEL_NAME, MODELS_FINAL_DIR


def load_model(
    model_path: Union[str, Path] = MODELS_FINAL_DIR / FINAL_MODEL_NAME,
) -> object:
    """
    Carrega o modelo definitivo treinado a partir de arquivo serializado (.joblib).
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo final não encontrado em: {path.resolve()}")
    return joblib.load(path)


def predict_risk(
    model: object,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Realiza a predição binária do nível de risco (0: Baixo, 1: Alto).
    """
    return model.predict(X)


def predict_risk_proba(
    model: object,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Estima as probabilidades calibradas de risco [P(Baixo), P(Alto)].
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError("O modelo fornecido não implementa estimação de probabilidades (predict_proba).")
    return model.predict_proba(X)


def format_risk_prediction(
    model: object,
    X_single_row: pd.DataFrame,
) -> Dict[str, Union[str, float]]:
    """
    Formata o resultado da inferência para uso amigável pela comunidade acadêmica.
    """
    pred_class = int(model.predict(X_single_row)[0])
    label = "Risco Alto" if pred_class == 1 else "Risco Baixo"

    prob_high = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_single_row)[0]
        prob_high = float(probs[1])

    return {
        "predicao_classe": pred_class,
        "classificacao_risco": label,
        "probabilidade_risco_alto": prob_high,
    }
