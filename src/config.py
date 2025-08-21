# src/config.py

# --- File Paths ---
DATA_PATH = "data/cumulative.csv"

# --- Modeling Constants ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "koi_disposition"

# --- Feature Selection ---
COLS_TO_DROP = [
    'rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition',
    'koi_score', 'koi_tce_delivname', 'koi_teq_err1', 'koi_teq_err2'
]
