"""
Script utilitário para reprocessar o dataset bruto gerando dados_processados.csv
com todas as variáveis temporais dinâmicas, tratamento geodésico e consistência metodológica.
"""

from datetime import date
from pathlib import Path
import sys

# Adiciona o diretório raiz ao PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from modules.config import (
    ANALYSIS_RADIUS_KM,
    CONSOLIDATED_DATASET_NAME,
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    IFSP_SPO_COORDINATES,
    INITIAL_YEAR,
    PROCESSED_DATASET_NAME,
)
from modules.features import (
    aggregate_crime_counts,
    build_cartesian_grid,
    engineer_features,
    filter_by_radius,
    map_period,
)


def main():
    print("Iniciando reprocessamento dos dados...")
    interim_path = DATA_INTERIM_DIR / CONSOLIDATED_DATASET_NAME
    if not interim_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {interim_path}")

    desired_cols = [
        "NUM_BO",
        "BAIRRO",
        "LATITUDE",
        "LONGITUDE",
        "DATA_COMUNICACAO_BO",
        "DATA_OCORRENCIA_BO",
        "HORA_OCORRENCIA_BO",
        "DESCR_PERIODO",
        "RUBRICA",
        "ano_referencia",
    ]

    print("Carregando dataset consolidado...")
    df = pd.read_csv(interim_path, sep=";", usecols=desired_cols, low_memory=False)
    print(f"Total bruto de registros lidos: {len(df)}")

    # Tipagem e limpeza inicial
    df["DATA_OCORRENCIA_BO"] = pd.to_datetime(df["DATA_OCORRENCIA_BO"], errors="coerce")
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")

    # Filtragem temporal >= INITIAL_YEAR (2022) e coordenadas válidas
    start_date = date(INITIAL_YEAR, 1, 1)
    end_date = df["DATA_OCORRENCIA_BO"].dt.date.max()

    mask_universe = (
        (df["DATA_OCORRENCIA_BO"].dt.date >= start_date)
        & df["LATITUDE"].notna()
        & df["LONGITUDE"].notna()
    )
    df_filtered = df[mask_universe].drop_duplicates(subset=["NUM_BO"]).reset_index(drop=True)

    print(f"Registros pós-desduplicação e filtro temporal inicial: {len(df_filtered)}")

    # Filtragem geodésica (raio de 2.0 km ao redor do IFSP)
    df_filtered = filter_by_radius(
        df_filtered,
        lat_col="LATITUDE",
        lon_col="LONGITUDE",
        center_lat=IFSP_SPO_COORDINATES[0],
        center_lon=IFSP_SPO_COORDINATES[1],
        radius_km=ANALYSIS_RADIUS_KM,
        include_distance=True,
    )
    print(f"Registros dentro do raio de {ANALYSIS_RADIUS_KM} km: {len(df_filtered)}")

    # Engenharia do período do dia (com fallback)
    df_filtered["DATA_HORA_OCORRENCIA_BO"] = pd.to_datetime(
        df_filtered["DATA_OCORRENCIA_BO"].astype(str) + " " + df_filtered["HORA_OCORRENCIA_BO"].astype(str),
        errors="coerce",
    )
    df_filtered["DESCR_PERIODO_OCORRENCIA_BO"] = df_filtered.apply(map_period, axis=1)

    n_with_period = df_filtered["DESCR_PERIODO_OCORRENCIA_BO"].notna().sum()
    n_without_period = df_filtered["DESCR_PERIODO_OCORRENCIA_BO"].isna().sum()
    print(f"Ocorrências com período mapeável: {n_with_period} ({n_with_period/len(df_filtered)*100:.1f}%)")
    print(f"Ocorrências sem período (hora incerta): {n_without_period} ({n_without_period/len(df_filtered)*100:.1f}%)")

    # Grid cartesiano completo e agregação com fillna(0)
    grid = build_cartesian_grid(start_date, end_date)
    df_agg = aggregate_crime_counts(df_filtered, grid)
    print(f"Grid cartesiano completo gerado com {len(df_agg)} janelas temporais de 6 horas.")

    # Engenharia de variáveis temporais dinâmicas (lags, médias móveis causais, one-hot)
    df_processed = engineer_features(df_agg)

    # Salvamento de dados_processados.csv
    out_path = DATA_PROCESSED_DIR / PROCESSED_DATASET_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(out_path, sep=";", index=False, encoding="utf-8")
    print(f"Dataset processado final exportado com sucesso em: {out_path} ({len(df_processed)} linhas, {len(df_processed.columns)} colunas).")


if __name__ == "__main__":
    main()
