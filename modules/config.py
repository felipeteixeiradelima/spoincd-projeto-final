"""
Configurações globais e hiperparâmetros do projeto SPOINCD.
"""

from pathlib import Path

# Diretórios base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
MODELS_CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
MODELS_FINAL_DIR = MODELS_DIR / "final"

# Nomes de arquivos
CONSOLIDATED_DATASET_NAME = "SPDadosCriminais_Unificado.csv"
PROCESSED_DATASET_NAME = "dados_processados.csv"
FINAL_MODEL_NAME = "modelo_risco_ifsp_final.joblib"
LOG_EXPERIMENTS_NAME = "log_experimentos.csv"

# Coordenadas geodésicas do campus IFSP - São Paulo (Rua Pedro Vicente, 625)
IFSP_SPO_COORDINATES = (-23.5237759, -46.624897)

# Raio de análise em quilômetros
ANALYSIS_RADIUS_KM = 2.0

# Raio médio da Terra em km (WGS-84 / IUGG)
EARTH_RADIUS_KM = 6371.0088

# Parâmetros temporais
INITIAL_YEAR = 2022
PERIOD_BLOCKS = ["MADRUGADA", "MANHÃ", "TARDE", "NOITE"]

# Parâmetros de particionamento e reprodutibilidade
DEV_TEST_SPLIT_RATIO = 0.85
N_EXPERIMENT_ITERATIONS = 30
BASE_RANDOM_SEED = 42

# Hiperparâmetros base dos classificadores supervisionados
LOGISTIC_REGRESSION_MAX_ITER = 500
DECISION_TREE_MAX_DEPTH = 6
RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_FOREST_MAX_DEPTH = 6
GRADIENT_BOOSTING_N_ESTIMATORS = 100
GRADIENT_BOOSTING_MAX_DEPTH = 3
