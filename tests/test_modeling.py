"""
Testes unitários para o módulo de modelagem, sanidade e testes estatísticos formais.
"""

import unittest

import numpy as np
import pandas as pd

from modules.modeling.evaluation import (
    compute_confidence_intervals,
    run_friedman_test,
    run_posthoc_wilcoxon,
)
from modules.modeling.train import get_model_constructors, run_sanity_check


class TestModeling(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Amostra sintética de 30 instâncias para testes de sanidade
        self.X_sample = pd.DataFrame({
            "MES": np.random.randint(1, 13, size=30),
            "DIA_SEMANA": np.random.randint(0, 7, size=30),
            "FLAG_FIM_DE_SEMANA": np.random.randint(0, 2, size=30),
            "PERIODO_MADRUGADA": [1, 0, 0, 0] * 7 + [1, 0],
            "PERIODO_MANHÃ": [0, 1, 0, 0] * 7 + [0, 1],
            "PERIODO_TARDE": [0, 0, 1, 0] * 7 + [0, 0],
            "PERIODO_NOITE": [0, 0, 0, 1] * 7 + [0, 0],
            "CRIMES_LAG_1": np.random.randint(0, 10, size=30),
            "CRIMES_LAG_4": np.random.randint(0, 10, size=30),
            "CRIMES_MEDIA_MOVEL_7D": np.random.uniform(1.0, 8.0, size=30),
            "PERIODO_LETIVO_IFSP": np.random.randint(0, 2, size=30),
        })
        self.y_sample = pd.Series(np.random.randint(0, 2, size=30))

    def test_model_constructors_count(self):
        """O estudo deve comparar exatamente 4 famílias de modelos supervisionados."""
        models = get_model_constructors()
        self.assertEqual(len(models), 4)
        expected_names = {"Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"}
        self.assertEqual(set(models.keys()), expected_names)

    def test_sanity_check_all_four_models(self):
        """O teste de sanidade deve validar convergência e consistência de todos os 4 modelos."""
        models = get_model_constructors()
        accuracies = run_sanity_check(models, self.X_sample, self.y_sample)
        self.assertEqual(len(accuracies), 4)
        # Árvore deve ser capaz de memorizar a amostra pequena (overfit de teste)
        self.assertGreater(accuracies["Decision Tree"], 0.95)

    def test_confidence_intervals(self):
        """Testa o cálculo dos intervalos de confiança a 95% para os modelos."""
        df_mock = pd.DataFrame({
            "Modelo": ["Model_A"] * 30 + ["Model_B"] * 30,
            "Seed": list(range(30)) * 2,
            "F1_Score": list(np.random.normal(0.70, 0.02, 30)) + list(np.random.normal(0.65, 0.02, 30)),
        })
        df_ci = compute_confidence_intervals(df_mock, metric="F1_Score")
        self.assertEqual(len(df_ci), 2)
        for _, row in df_ci.iterrows():
            self.assertLess(row["IC_95%_Inf"], row["Média"])
            self.assertGreater(row["IC_95%_Sup"], row["Média"])

    def test_friedman_and_posthoc(self):
        """Testa a execução combinada do teste omnibus de Friedman e post-hoc de Wilcoxon-Holm."""
        # Cria dados sintéticos pareados para 4 modelos
        seeds = list(range(30))
        records = []
        for s in seeds:
            records.append({"Modelo": "M1", "Seed": s, "F1_Score": 0.75 + np.random.normal(0, 0.01)})
            records.append({"Modelo": "M2", "Seed": s, "F1_Score": 0.74 + np.random.normal(0, 0.01)})
            records.append({"Modelo": "M3", "Seed": s, "F1_Score": 0.70 + np.random.normal(0, 0.01)})
            records.append({"Modelo": "M4", "Seed": s, "F1_Score": 0.60 + np.random.normal(0, 0.01)})

        df_exp = pd.DataFrame(records)

        stat, p_val, is_sig = run_friedman_test(df_exp)
        self.assertIsInstance(stat, float)
        self.assertIsInstance(p_val, float)
        self.assertTrue(is_sig)

        df_posthoc = run_posthoc_wilcoxon(df_exp)
        # 4 modelos -> 4 * 3 / 2 = 6 comparações pareadas
        self.assertEqual(len(df_posthoc), 6)
        self.assertIn("p_valor_ajustado_Holm", df_posthoc.columns)
        self.assertIn("Significante_5%", df_posthoc.columns)


if __name__ == "__main__":
    unittest.main()
