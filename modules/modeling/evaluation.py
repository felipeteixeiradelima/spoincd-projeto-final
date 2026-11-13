"""
Módulo de análise estatística formal (Friedman, Post-Hoc Wilcoxon-Holm, Intervalos de Confiança)
e avaliação rigorosa de generalização em dados não vistos.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_confidence_intervals(
    df_results: pd.DataFrame,
    metric: str = "F1_Score",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Passo 5 do Enunciado: Calcula Intervalos de Confiança paramétricos (distribuição t de Student)
    para cada classificador a partir da amostra de erros das múltiplas partições.
    """
    summary = []

    for model_name, group in df_results.groupby("Modelo"):
        vals = group[metric].to_numpy(dtype=float)
        n = len(vals)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))
        se = std / np.sqrt(n)

        # Graus de liberdade = n - 1
        t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
        median = float(np.median(vals))

        summary.append({
            "Modelo": model_name,
            "Amostras": n,
            "Média": mean,
            "Desvio_Padrão": std,
            "Erro_Padrão": se,
            f"IC_{int(confidence*100)}%_Inf": ci_lower,
            f"IC_{int(confidence*100)}%_Sup": ci_upper,
            "Mediana": median,
        })

    df_ci = pd.DataFrame(summary).sort_values(by="Média", ascending=False).reset_index(drop=True)
    return df_ci


def run_friedman_test(
    df_results: pd.DataFrame,
    metric: str = "F1_Score",
) -> Tuple[float, float, bool]:
    """
    Passo 5 do Enunciado: Executa o Teste Omnibus de Friedman para amostras pareadas (>2 grupos).
    Retorna a estatística de teste, o p-valor e o booleano de significância (p < 0.05).
    """
    pivot = df_results.pivot(index="Seed", columns="Modelo", values=metric)
    cols = pivot.columns.tolist()

    arrays = [pivot[col].to_numpy() for col in cols]
    stat, p_value = stats.friedmanchisquare(*arrays)

    is_significant = bool(p_value < 0.05)
    return float(stat), float(p_value), is_significant


def run_posthoc_wilcoxon(
    df_results: pd.DataFrame,
    metric: str = "F1_Score",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Passo 5 do Enunciado: Teste Post-Hoc formal de Wilcoxon pareado (Signed-Rank Test)
    com correção de Holm-Bonferroni para controle de erro de família (FWER).
    """
    pivot = df_results.pivot(index="Seed", columns="Modelo", values=metric)
    models = sorted(pivot.columns.tolist())

    comparisons = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            s1 = pivot[m1].to_numpy()
            s2 = pivot[m2].to_numpy()
            diff = s1 - s2

            # Teste de postos sinalizados de Wilcoxon
            stat, p_val = stats.wilcoxon(s1, s2)

            wins_m1 = int(np.sum(diff > 0))
            wins_m2 = int(np.sum(diff < 0))
            ties = int(np.sum(diff == 0))

            comparisons.append({
                "Comparação": f"{m1} vs {m2}",
                "Modelo_A": m1,
                "Modelo_B": m2,
                "Diferença_Média (A - B)": float(np.mean(diff)),
                "Vitórias_A": wins_m1,
                "Vitórias_B": wins_m2,
                "Empates": ties,
                "Estatística_W": float(stat),
                "p_valor_bruto": float(p_val),
            })

    df_comp = pd.DataFrame(comparisons)

    # Ordenação crescente por p-valor bruto para aplicação do método de Holm-Bonferroni
    df_comp = df_comp.sort_values(by="p_valor_bruto").reset_index(drop=True)
    m = len(df_comp)

    adjusted_p_values = []
    significant_flags = []

    for rank, p in enumerate(df_comp["p_valor_bruto"], start=1):
        # Fator de Holm: (m - rank + 1)
        factor = m - rank + 1
        adj_p = min(1.0, p * factor)
        adjusted_p_values.append(adj_p)
        significant_flags.append(adj_p < alpha)

    # Garantir monotonicidade dos p-valores ajustados
    for k in range(1, m):
        if adjusted_p_values[k] < adjusted_p_values[k - 1]:
            adjusted_p_values[k] = adjusted_p_values[k - 1]

    df_comp["p_valor_ajustado_Holm"] = adjusted_p_values
    df_comp["Significante_5%"] = [p < alpha for p in adjusted_p_values]

    return df_comp


def evaluate_unseen_data(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    """
    Passo 6 do Enunciado: Avaliação rigorosa da capacidade de generalização em dados não vistos.
    Compara o modelo treinado em X_dev com um baseline ingênuo da classe majoritária (DummyClassifier).
    """
    # Treinamento do modelo supervisionado no conjunto de desenvolvimento completo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Treinamento do baseline da classe mais frequente
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    # Cálculo das métricas
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    dummy_acc = float(accuracy_score(y_test, y_pred_dummy))
    dummy_f1 = float(f1_score(y_test, y_pred_dummy, zero_division=0))

    model_acc = float(accuracy_score(y_test, y_pred))
    model_f1 = float(f1_score(y_test, y_pred, zero_division=0))
    model_prec = float(precision_score(y_test, y_pred, zero_division=0))
    model_rec = float(recall_score(y_test, y_pred, zero_division=0))

    roc_auc = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, probs))

    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "model_metrics": {
            "Acurácia": model_acc,
            "F1_Score": model_f1,
            "Precisão_Classe_1": model_prec,
            "Recall_Classe_1": model_rec,
            "ROC_AUC": roc_auc,
        },
        "baseline_metrics": {
            "Acurácia": dummy_acc,
            "F1_Score": dummy_f1,
        },
    }


def analyze_concept_drift(
    df_dev: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "RISCO_ALTO",
    count_col: str = "QTD_CRIMES",
) -> Dict[str, float]:
    """
    Auditoria de Deslocamento de Conceito (Concept Drift) e Deslocamento de Covariáveis (Covariate Shift).
    Quantifica a discrepância entre a distribuição dos dados de treino e teste.
    """
    dev_total = len(df_dev)
    test_total = len(df_test)

    dev_crimes_mean = float(df_dev[count_col].mean())
    test_crimes_mean = float(df_test[count_col].mean())

    dev_high_risk_pct = float((df_dev[target_col].sum() / dev_total) * 100.0)
    test_high_risk_pct = float((df_test[target_col].sum() / test_total) * 100.0)

    drift_relative_pct = float(((test_crimes_mean - dev_crimes_mean) / dev_crimes_mean) * 100.0)

    return {
        "dev_instances": dev_total,
        "test_instances": test_total,
        "dev_crimes_mean_per_window": dev_crimes_mean,
        "test_crimes_mean_per_window": test_crimes_mean,
        "dev_high_risk_prevalence_pct": dev_high_risk_pct,
        "test_high_risk_prevalence_pct": test_high_risk_pct,
        "crime_volume_drift_pct": drift_relative_pct,
    }
