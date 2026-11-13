"""
Módulos de modelagem supervisionada, testes de sanidade, avaliação e inferência.
"""

from modules.modeling.train import get_model_constructors, run_experiments, run_sanity_check
from modules.modeling.evaluation import (
    compute_confidence_intervals,
    run_friedman_test,
    run_posthoc_wilcoxon,
    evaluate_unseen_data,
)

__all__ = [
    "get_model_constructors",
    "run_sanity_check",
    "run_experiments",
    "compute_confidence_intervals",
    "run_friedman_test",
    "run_posthoc_wilcoxon",
    "evaluate_unseen_data",
]
