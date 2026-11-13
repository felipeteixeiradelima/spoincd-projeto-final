"""
Testes unitários para o módulo de features, distâncias geodésicas e engenharia de dados.
"""

from datetime import date
import unittest

import numpy as np
import pandas as pd

from modules.config import EARTH_RADIUS_KM, IFSP_SPO_COORDINATES
from modules.features import (
    apply_risk_threshold_without_leakage,
    build_cartesian_grid,
    calculate_haversine_distance,
    engineer_features,
    filter_by_radius,
    map_period,
    split_dev_test_chronological,
)


class TestFeatures(unittest.TestCase):
    def test_haversine_zero_distance(self):
        """A distância de um ponto para ele mesmo deve ser 0.0."""
        lat, lon = IFSP_SPO_COORDINATES
        d = calculate_haversine_distance(lat, lon, lat, lon)
        self.assertAlmostEqual(d, 0.0, places=4)

    def test_haversine_known_distance(self):
        """A distância entre IFSP e Praça da Sé deve ser ~2.5 km."""
        lat_ifsp, lon_ifsp = IFSP_SPO_COORDINATES
        lat_se, lon_se = -23.5505, -46.6333
        d = calculate_haversine_distance(lat_ifsp, lon_ifsp, lat_se, lon_se)
        self.assertTrue(2.0 < d < 4.0, f"Distância esperada ~3 km, obteve: {d:.2f}")

    def test_filter_by_radius(self):
        """Verifica se apenas registros dentro do raio de 2km são mantidos."""
        lat_ifsp, lon_ifsp = IFSP_SPO_COORDINATES
        data = {
            "LATITUDE": [lat_ifsp, lat_ifsp + 0.005, -23.65],  # O 3º ponto é bem longe (~14 km)
            "LONGITUDE": [lon_ifsp, lon_ifsp + 0.005, -46.65],
            "NUM_BO": ["BO1", "BO2", "BO3"],
        }
        df = pd.DataFrame(data)
        filtered = filter_by_radius(df, radius_km=2.0)
        self.assertEqual(len(filtered), 2)
        self.assertIn("distancia_km", filtered.columns)
        self.assertTrue((filtered["distancia_km"] <= 2.0).all())

    def test_map_period_by_hour(self):
        """Testa o mapeamento correto dos 4 blocos horários."""
        times = [
            (pd.Timestamp("2023-05-10 03:00:00"), "MADRUGADA"),
            (pd.Timestamp("2023-05-10 09:30:00"), "MANHÃ"),
            (pd.Timestamp("2023-05-10 14:15:00"), "TARDE"),
            (pd.Timestamp("2023-05-10 21:45:00"), "NOITE"),
        ]
        for dt, expected in times:
            row = pd.Series({"DATA_HORA_OCORRENCIA_BO": dt, "DESCR_PERIODO": np.nan})
            self.assertEqual(map_period(row), expected)

    def test_map_period_fallback(self):
        """Testa o fallback para a descrição textual quando a hora for ausente."""
        row_fallback = pd.Series({
            "DATA_HORA_OCORRENCIA_BO": pd.NaT,
            "DESCR_PERIODO": "PELA MANHÃ",
        })
        self.assertEqual(map_period(row_fallback), "MANHÃ")

        row_unknown = pd.Series({
            "DATA_HORA_OCORRENCIA_BO": pd.NaT,
            "DESCR_PERIODO": "EM HORA INCERTA",
        })
        self.assertTrue(pd.isna(map_period(row_unknown)))

    def test_build_cartesian_grid(self):
        """Testa se o grid cartesiano gera exatamente 4 blocos por dia."""
        d_start = date(2023, 1, 1)
        d_end = date(2023, 1, 10)  # 10 dias
        grid = build_cartesian_grid(d_start, d_end)
        self.assertEqual(len(grid), 10 * 4)

    def test_risk_threshold_without_leakage(self):
        """Verifica que a mediana calculada no treino não vaza do conjunto de teste."""
        df_dev = pd.DataFrame({"QTD_CRIMES": [1, 2, 3, 4, 10]})  # Mediana = 3.0
        df_test = pd.DataFrame({"QTD_CRIMES": [1, 2, 3, 4, 100, 200]})  # Mediana do teste seria bem maior

        dev_res, test_res, thresh = apply_risk_threshold_without_leakage(df_dev, df_test)

        self.assertEqual(thresh, 3.0)
        # No dev: > 3.0 -> [0, 0, 0, 1, 1]
        self.assertListEqual(dev_res["RISCO_ALTO"].tolist(), [0, 0, 0, 1, 1])
        # No test com limiar 3.0: [0, 0, 0, 1, 1, 1]
        self.assertListEqual(test_res["RISCO_ALTO"].tolist(), [0, 0, 0, 1, 1, 1])

    def test_causal_lags_no_lookahead(self):
        """Garante que as variáveis de lag e média móvel não olham para o turno atual."""
        df_agg = pd.DataFrame({
            "DATA": pd.date_range("2023-01-01", periods=10, freq="D").repeat(4).date,
            "DESCR_PERIODO_OCORRENCIA_BO": ["MADRUGADA", "MANHÃ", "TARDE", "NOITE"] * 10,
            "QTD_CRIMES": list(range(40)),
        })
        df_feat = engineer_features(df_agg)

        # Na primeira linha, os lags devem ser 0 (não há passado)
        self.assertEqual(df_feat["CRIMES_LAG_1"].iloc[0], 0)
        # Na segunda linha, o lag 1 deve ser o valor da primeira (0)
        self.assertEqual(df_feat["CRIMES_LAG_1"].iloc[1], 0)
        # Na terceira linha, o lag 1 deve ser o valor da segunda (1)
        self.assertEqual(df_feat["CRIMES_LAG_1"].iloc[2], 1)


if __name__ == "__main__":
    unittest.main()
