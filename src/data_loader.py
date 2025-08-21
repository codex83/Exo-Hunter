# src/data_loader.py
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Loads data from a CSV file into a pandas DataFrame."""
    print(f"Loading data from {path}...")
    try:
        df = pd.read_csv(path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return None
