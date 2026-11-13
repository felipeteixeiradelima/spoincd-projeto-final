"""
Treinamento de modelos, testes de sanidade e rotina de validação experimental com sementes.
"""

from pathlib import Path
from typing import Callable, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from modules.config import (
    BASE_RANDOM_SEED,
    DECISION_TREE_MAX_DEPTH,
    GRADIENT_BOOSTING_MAX_DEPTH,
    GRADIENT_BOOSTING_N_ESTIMATORS,
    LOG_EXPERIMENTS_NAME,
    LOGISTIC_REGRESSION_MAX_ITER,
    MODELS_CHECKPOINTS_DIR,
    MODELS_FINAL_DIR,
    N_EXPERIMENT_ITERATIONS,
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_N_ESTIMATORS,
)


def get_model_constructors() -> Dict[str, Callable[[int], object]]:
    """
    Retorna o catálogo dos 4 classificadores supervisionados comparados no estudo,
    parametrizados para receber a semente do gerador aleatório.
    A Regressão Logística inclui padronização (StandardScaler) para garantir convergência.
    """
    return {
        "Logistic Regression": lambda s: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                random_state=s,
                max_iter=LOGISTIC_REGRESSION_MAX_ITER,
            ),
        ),
        "Decision Tree": lambda s: DecisionTreeClassifier(
            random_state=s,
            max_depth=DECISION_TREE_MAX_DEPTH,
        ),
        "Random Forest": lambda s: RandomForestClassifier(
            random_state=s,
            n_estimators=RANDOM_FOREST_N_ESTIMATORS,
            max_depth=RANDOM_FOREST_MAX_DEPTH,
        ),
        "Gradient Boosting": lambda s: GradientBoostingClassifier(
            random_state=s,
            n_estimators=GRADIENT_BOOSTING_N_ESTIMATORS,
            max_depth=GRADIENT_BOOSTING_MAX_DEPTH,
        ),
    }


def run_sanity_check(
    models_dict: Dict[str, Callable[[int], object]],
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
) -> Dict[str, float]:
    """
    Passo 1 do Enunciado: Teste de Sanidade e Consistência.
    Valida se todos os 4 modelos implementados convergem, verificando:
    1. Consistência de dimensões de entrada (X) e saída (y_pred).
    2. Ausência de valores nulos (NaN/Inf) nos dados e predições.
    3. Capacidade de ajuste em subconjunto mínimo.
    """
    assert len(X_sample) == len(y_sample), "Inconsistência de dimensões entre X e y na sanidade."
    assert not X_sample.isna().any().any(), "Presença de valores nulos nas features de sanidade."
    assert not y_sample.isna().any(), "Presença de valores nulos no target de sanidade."

    n_samples = len(X_sample)
    accuracies = {}

    for name, factory in models_dict.items():
        # Para a árvore no teste de sobreajuste proposital, usamos sem limite de profundidade
        if name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=BASE_RANDOM_SEED)
        else:
            model = factory(BASE_RANDOM_SEED)

        model.fit(X_sample, y_sample)
        preds = model.predict(X_sample)

        # Checagens estruturais de consistência
        assert len(preds) == n_samples, f"Modelo {name} retornou número incorreto de predições."
        assert set(np.unique(preds)).issubset({0, 1}), f"Modelo {name} retornou classes fora de {{0, 1}}."

        # Checagem de probabilidades quando disponível
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_sample)
            assert probs.shape == (n_samples, 2), f"Formato incorreto de probabilidades para {name}."
            assert np.allclose(probs.sum(axis=1), 1.0), f"Soma de probabilidades != 1.0 para {name}."

        acc = float(accuracy_score(y_sample, preds))
        accuracies[name] = acc

    # A árvore sem limite de profundidade deve ser capaz de memorizar o subconjunto (overfit intencional)
    assert accuracies["Decision Tree"] > 0.95, (
        f"Falha de sanidade na Decision Tree: acurácia {accuracies['Decision Tree']:.2f} <= 0.95"
    )

    return accuracies


def run_experiments(
    models_dict: Dict[str, Callable[[int], object]],
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    n_iterations: int = N_EXPERIMENT_ITERATIONS,
    base_seed: int = BASE_RANDOM_SEED,
    val_size: float = 0.20,
    checkpoints_dir: Path = MODELS_CHECKPOINTS_DIR,
    log_name: str = LOG_EXPERIMENTS_NAME,
) -> pd.DataFrame:
    """
    Passos 3 e 4 do Enunciado: Execução e Replicação Experimental.
    Executa a validação cruzada Monte Carlo estratificada em n_iterations partições com sementes distintas.
    Registra métricas (F1, Acurácia, Precisão, Recall) e salva pesos intermediários (.pkl).
    """
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for i in range(n_iterations):
        seed = base_seed + i
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev, test_size=val_size, random_state=seed, stratify=y_dev
        )

        for model_name, factory in models_dict.items():
            model = factory(seed)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            f1 = float(f1_score(y_val, preds, zero_division=0))
            acc = float(accuracy_score(y_val, preds))
            prec = float(precision_score(y_val, preds, zero_division=0))
            rec = float(recall_score(y_val, preds, zero_division=0))

            records.append({
                "Modelo": model_name,
                "Seed": seed,
                "F1_Score": f1,
                "Acuracia": acc,
                "Precisao": prec,
                "Recall": rec,
            })

            # Salvar checkpoint serializado
            clean_name = model_name.lower().replace(" ", "_")
            joblib.dump(model, checkpoints_dir / f"{clean_name}seed{seed}.pkl")

    df_results = pd.DataFrame(records)
    df_results.to_csv(checkpoints_dir / log_name, index=False)
    return df_results


def train_final_model(
    model_factory: Callable[[int], object],
    X_full: pd.DataFrame,
    y_full: pd.Series,
    seed: int = BASE_RANDOM_SEED,
    export_path: Path = MODELS_FINAL_DIR / "modelo_risco_ifsp_final.joblib",
) -> object:
    """
    Passo 7 do Enunciado: Treinamento da versão final do modelo.
    Utiliza todos os dados disponíveis (desenvolvimento + teste) para gerar o artefato de deploy.
    """
    export_path.parent.mkdir(parents=True, exist_ok=True)
    final_model = model_factory(seed)
    final_model.fit(X_full, y_full)
    joblib.dump(final_model, export_path)
    return final_model
