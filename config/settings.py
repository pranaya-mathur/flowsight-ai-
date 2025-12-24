from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURE_STORE_DIR = BASE_DIR / "data" / "feature_store"
MODELS_DIR = BASE_DIR / "models"

PRIMARY_DATA_PATH = DATA_RAW_DIR / "supply_chain_15000_expanded.csv"
VENDOR_DATA_PATH = DATA_RAW_DIR / "vendor_performance_download.csv"
ROUTE_DATA_PATH = DATA_RAW_DIR / "shipments_augmented.csv"

ENSEMBLE_MODEL_PATH = MODELS_DIR / "ensemble_delay_model.pkl"
MULTITASK_MODEL_PATH = MODELS_DIR / "multitask_delay_model.pt"

LOG_LEVEL = "INFO"
