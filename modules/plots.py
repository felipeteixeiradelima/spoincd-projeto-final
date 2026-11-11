"""
Módulo de geração de gráficos e visualizações analíticas para AED e resultados de modelos.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_academic_plot_style():
    """
    Define a identidade visual acadêmica para figuras de publicação (formato ACL / SBBD).
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_f1_distributions(
    df_results: pd.DataFrame,
    metric: str = "F1_Score",
    title: str = "Distribuição de Desempenho nas 30 Partições (Validação Monte Carlo)",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Gera gráfico de boxplot comparativo com distribuição das métricas dos 4 modelos.
    """
    set_academic_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df_results,
        x="Modelo",
        y=metric,
        palette="Set2",
        ax=ax,
        width=0.45,
        boxprops=dict(alpha=0.85),
    )
    sns.stripplot(
        data=df_results,
        x="Modelo",
        y=metric,
        color="black",
        alpha=0.35,
        jitter=0.15,
        size=5,
        ax=ax,
    )

    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Modelo Supervisionado", fontsize=11, labelpad=8)
    ax.set_ylabel(metric.replace("_", " "), fontsize=11, labelpad=8)

    fig.tight_layout()
    return fig


def plot_confidence_intervals(
    df_ci: pd.DataFrame,
    metric_label: str = "F1-Score",
    title: str = "Intervalos de Confiança a 95% para os Modelos Supervisionados",
    figsize: tuple = (9, 5),
) -> plt.Figure:
    """
    Gera gráfico com pontos médios e barras de erro de 95% de confiança.
    """
    set_academic_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    y_err_lower = df_ci["Média"] - df_ci["IC_95%_Inf"]
    y_err_upper = df_ci["IC_95%_Sup"] - df_ci["Média"]
    y_err = [y_err_lower, y_err_upper]

    x_pos = np.arange(len(df_ci))
    ax.errorbar(
        x_pos,
        df_ci["Média"],
        yerr=y_err,
        fmt="o",
        color="#2b5c8f",
        ecolor="#d95f02",
        elinewidth=2.2,
        capsize=6,
        capthick=1.5,
        markersize=8,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_ci["Modelo"], fontsize=10)
    ax.set_title(title, fontsize=12, weight="bold", pad=10)
    ax.set_ylabel(f"Média {metric_label} (com IC 95%)", fontsize=11)
    ax.set_xlabel("Modelo", fontsize=11)

    for i, row in df_ci.iterrows():
        ax.annotate(
            f"{row['Média']:.4f}\n[{row['IC_95%_Inf']:.4f}, {row['IC_95%_Sup']:.4f}]",
            (i, row["Média"]),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.9),
        )

    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str = "Random Forest",
    class_names: Optional[List[str]] = None,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """
    Gera mapa de calor da matriz de confusão com anotações de contagem.
    """
    set_academic_plot_style()
    if class_names is None:
        class_names = ["Baixo (0)", "Alto (1)"]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=False,
    )

    ax.set_title(f"Matriz de Confusão em Dados Não Vistos ({model_name})", fontsize=11, weight="bold", pad=10)
    ax.set_xlabel("Risco Previsto", fontsize=10)
    ax.set_ylabel("Risco Real", fontsize=10)

    fig.tight_layout()
    return fig
