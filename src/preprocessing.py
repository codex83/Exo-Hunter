# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def preprocess_data(df: pd.DataFrame, cols_to_drop: list, target_column: str) -> (pd.DataFrame, pd.DataFrame):
    """Performs all preprocessing steps on the raw dataframe."""
    print("Starting data preprocessing...")

    # --- Separate Datasets ---
    df_primary = df[df[target_column].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    df_candidates = df[df[target_column] == 'CANDIDATE'].copy()
    print(f"Separated into primary ({df_primary.shape}) and candidate ({df_candidates.shape}) datasets.")

    # --- Feature Selection ---
    df_primary.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df_candidates.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"Dropped specified columns.")

    # --- Impute Missing Values ---
    imputer = SimpleImputer(strategy='median')
    numeric_cols = df_primary.select_dtypes(include=np.number).columns.tolist()
    # Ensure target column is not in numeric_cols if it's already encoded
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
        
    df_primary[numeric_cols] = imputer.fit_transform(df_primary[numeric_cols])
    candidate_numeric_cols = df_candidates.select_dtypes(include=np.number).columns.tolist()
    df_candidates[candidate_numeric_cols] = imputer.transform(df_candidates[candidate_numeric_cols])
    print("Imputed missing values using median strategy.")

    # --- Feature Engineering: SNR ---
    error_cols = [col for col in df_primary.columns if 'err' in col]
    measurement_cols = sorted(list(set([col.replace('_err1', '').replace('_err2', '') for col in error_cols])))
    
    for col in measurement_cols:
        err1_col, err2_col = f"{col}_err1", f"{col}_err2"
        if err1_col in df_primary.columns and err2_col in df_primary.columns:
            # For primary df
            avg_error_primary = np.sqrt(df_primary[err1_col]**2 + df_primary[err2_col]**2)
            df_primary[f'{col}_snr'] = df_primary[col] / avg_error_primary
            # For candidates df
            avg_error_candidates = np.sqrt(df_candidates[err1_col]**2 + df_candidates[err2_col]**2)
            df_candidates[f'{col}_snr'] = df_candidates[col] / avg_error_candidates
    
    df_primary.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_candidates.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    snr_cols = [col for col in df_primary.columns if '_snr' in col]
    if snr_cols:
        df_primary[snr_cols] = imputer.fit_transform(df_primary[snr_cols])
        df_candidates[snr_cols] = imputer.transform(df_candidates[snr_cols])

    df_primary.drop(columns=error_cols, inplace=True, errors='ignore')
    df_candidates.drop(columns=error_cols, inplace=True, errors='ignore')
    print("Engineered SNR features and dropped original error columns.")

    # --- Target Encoding ---
    target_map = {'CONFIRMED': 1, 'FALSE POSITIVE': 0}
    df_primary[target_column] = df_primary[target_column].map(target_map)
    print("Encoded target variable.")
    
    print("Preprocessing complete.")
    return df_primary, df_candidates
