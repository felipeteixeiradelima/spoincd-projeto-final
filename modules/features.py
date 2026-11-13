"""
Módulo de engenharia de variáveis, filtragem espacial e agregação temporal.
"""

from datetime import date
from typing import Tuple

import numpy as np
import pandas as pd

from modules.config import ANALYSIS_RADIUS_KM, EARTH_RADIUS_KM, IFSP_SPO_COORDINATES, PERIOD_BLOCKS


def calculate_haversine_distance(
    lat1: float | np.ndarray | pd.Series,
    lon1: float | np.ndarray | pd.Series,
    lat2: float | np.ndarray | pd.Series,
    lon2: float | np.ndarray | pd.Series,
    radius_km: float = EARTH_RADIUS_KM,
) -> np.ndarray | pd.Series:
    """
    Calcula a distância geodésica entre coordenadas usando a fórmula de Haversine vetorizada.
    """
    phi1 = np.radians(lat1)
    lambda1 = np.radians(lon1)
    phi2 = np.radians(lat2)
    lambda2 = np.radians(lon2)

    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    return radius_km * c


def filter_by_radius(
    df: pd.DataFrame,
    lat_col: str = "LATITUDE",
    lon_col: str = "LONGITUDE",
    center_lat: float = IFSP_SPO_COORDINATES[0],
    center_lon: float = IFSP_SPO_COORDINATES[1],
    radius_km: float = ANALYSIS_RADIUS_KM,
    include_distance: bool = True,
) -> pd.DataFrame:
    """
    Filtra linhas de um DataFrame dentro de um raio geodésico a partir de um ponto central.
    """
    valid_coords = df[lat_col].notna() & df[lon_col].notna()
    subset = df[valid_coords].copy()

    distances = calculate_haversine_distance(
        center_lat,
        center_lon,
        subset[lat_col].to_numpy(dtype=float),
        subset[lon_col].to_numpy(dtype=float),
    )

    mask = distances <= radius_km
    result = subset[mask].copy()

    if include_distance:
        result["distancia_km"] = distances[mask]

    return result


def map_period(row: pd.Series) -> str | float:
    """
    Mapeia a hora de ocorrência para 4 turnos de 6 horas, com mecanismo
    de contingência (fallback) para a descrição textual original da SSP-SP.
    """
    dt = row.get("DATA_HORA_OCORRENCIA_BO")
    if pd.notna(dt):
        h = dt.hour
        if 0 <= h < 6:
            return "MADRUGADA"
        if 6 <= h < 12:
            return "MANHÃ"
        if 12 <= h < 18:
            return "TARDE"
        if 18 <= h <= 23:
            return "NOITE"

    # Fallback para o texto registrado em DESCR_PERIODO
    desc = str(row.get("DESCR_PERIODO", "")).strip().upper()
    if "MADR" in desc:
        return "MADRUGADA"
    if "MANH" in desc:
        return "MANHÃ"
    if "TARD" in desc:
        return "TARDE"
    if "NOIT" in desc:
        return "NOITE"

    return np.nan


def build_cartesian_grid(
    start_date: date,
    end_date: date,
    periods: list[str] = PERIOD_BLOCKS,
) -> pd.DataFrame:
    """
    Constrói o grid cartesiano completo entre todas as datas e os períodos do dia,
    evitando que janelas com zero crimes sejam omitidas da amostra.
    """
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D").date
    grid = (
        pd.MultiIndex.from_product(
            [all_dates, periods], names=["DATA", "DESCR_PERIODO_OCORRENCIA_BO"]
        )
        .to_frame()
        .reset_index(drop=True)
    )
    return grid


def aggregate_crime_counts(
    df_filtered: pd.DataFrame,
    grid: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega as ocorrências criminais válidas por data e período, unindo ao grid cartesiano
    e imputando zero nas janelas de calmaria (sem ocorrências).
    """
    valid_occurrences = df_filtered.dropna(subset=["DESCR_PERIODO_OCORRENCIA_BO"]).copy()
    valid_occurrences["DATA"] = valid_occurrences["DATA_OCORRENCIA_BO"].dt.date

    counts = (
        valid_occurrences.groupby(["DATA", "DESCR_PERIODO_OCORRENCIA_BO"])["NUM_BO"]
        .count()
        .reset_index()
        .rename(columns={"NUM_BO": "QTD_CRIMES"})
    )

    merged = pd.merge(
        grid,
        counts,
        on=["DATA", "DESCR_PERIODO_OCORRENCIA_BO"],
        how="left",
    )
    merged["QTD_CRIMES"] = merged["QTD_CRIMES"].fillna(0).astype(int)

    return merged


def engineer_features(df_aggregated: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a engenharia de atributos: variáveis de calendário, defasagens causais
    (lags e médias móveis sem vazamento de futuro) e codificação one-hot dos turnos.
    """
    df = df_aggregated.copy()
    df["DATA"] = pd.to_datetime(df["DATA"])

    # Ordenação cronológica rigorosa para variáveis de defasagem (lags)
    period_order = {"MADRUGADA": 0, "MANHÃ": 1, "TARDE": 2, "NOITE": 3}
    df["_period_order"] = df["DESCR_PERIODO_OCORRENCIA_BO"].map(period_order)
    df = df.sort_values(["DATA", "_period_order"]).reset_index(drop=True)
    df = df.drop(columns=["_period_order"])

    # Variáveis de calendário
    df["ANO"] = df["DATA"].dt.year
    df["MES"] = df["DATA"].dt.month
    df["DIA_SEMANA"] = df["DATA"].dt.day_of_week
    df["FLAG_FIM_DE_SEMANA"] = df["DIA_SEMANA"].isin([5, 6]).astype(int)
    df["TRIMESTRE"] = df["DATA"].dt.quarter

    # Indicador de calendário letivo no campus IFSP-SPO:
    # Aulas regulares ocorrem tipicamente de fevereiro a junho (1º sem.) e agosto a dezembro (2º sem.),
    # com recesso/férias discentes em janeiro e julho.
    df["PERIODO_LETIVO_IFSP"] = df["MES"].isin([2, 3, 4, 5, 6, 8, 9, 10, 11, 12]).astype(int)

    # Features temporais dinâmicas e causais (estritamente passadas - sem look-ahead bias):
    # LAG 1: Volume de crimes no turno imediatamente anterior (t - 1)
    df["CRIMES_LAG_1"] = df["QTD_CRIMES"].shift(1).fillna(0).astype(int)

    # LAG 4: Volume de crimes no mesmo turno do dia anterior (t - 4)
    df["CRIMES_LAG_4"] = df["QTD_CRIMES"].shift(4).fillna(0).astype(int)

    # Média móvel causal de 7 dias (28 turnos anteriores): shift(1) garante que o turno atual não entra no cálculo
    df["CRIMES_MEDIA_MOVEL_7D"] = (
        df["QTD_CRIMES"].shift(1).rolling(window=28, min_periods=1).mean().fillna(0).round(2)
    )

    # One-Hot Encoding do período
    df = pd.get_dummies(
        df,
        prefix="PERIODO",
        columns=["DESCR_PERIODO_OCORRENCIA_BO"],
        dtype=int,
    )

    return df


def split_dev_test_chronological(
    df: pd.DataFrame,
    split_ratio: float = 0.85,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Realiza o particionamento temporal estrito: os primeiros split_ratio dos dados
    são destinados ao desenvolvimento e os (1 - split_ratio) finais ao teste de generalização.
    """
    df_sorted = df.sort_values(["DATA"]).reset_index(drop=True)
    split_idx = int(len(df_sorted) * split_ratio)
    df_dev = df_sorted.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df_sorted.iloc[split_idx:].copy().reset_index(drop=True)
    return df_dev, df_test


def apply_risk_threshold_without_leakage(
    df_dev: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Calcula a mediana estritamente sobre a partição de desenvolvimento (df_dev)
    e a aplica em ambas as partições, eliminando 100% de Target/Data Leakage.
    """
    threshold = float(df_dev["QTD_CRIMES"].median())

    dev_res = df_dev.copy()
    test_res = df_test.copy()

    dev_res["RISCO_ALTO"] = (dev_res["QTD_CRIMES"] > threshold).astype(int)
    test_res["RISCO_ALTO"] = (test_res["QTD_CRIMES"] > threshold).astype(int)

    return dev_res, test_res, threshold
