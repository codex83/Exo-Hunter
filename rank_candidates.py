# rank_candidates.py
import pandas as pd
import joblib
from src import config
from src.data_loader import load_data

def rank_candidates(model_path: str, data_path: str):
    """Loads a trained model and uses it to rank candidate exoplanets."""
    print("--- Starting Candidate Ranking ---")
    
    # Load the trained model pipeline
    try:
        model_pipeline = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please run main.py to train and save the model.")
        return

    # Load the raw data
    raw_df = load_data(data_path)
    if raw_df is None:
        return

    # Isolate the candidate data
    candidates_df = raw_df[raw_df[config.TARGET_COLUMN] == 'CANDIDATE'].copy()
    candidate_ids = candidates_df['kepoi_name']
    
    # The pipeline handles all preprocessing, so we pass the dataframe as is
    print(f"Predicting probabilities for {len(candidates_df)} candidates...")
    candidate_probs = model_pipeline.predict_proba(candidates_df)[:, 1]
    
    # Create and save the results
    results_df = pd.DataFrame({
        'kepoi_name': candidate_ids,
        'confirmation_probability': candidate_probs
    }).sort_values(by='confirmation_probability', ascending=False)
    
    output_path = "ranked_candidates.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\nTop 15 most likely exoplanet candidates:")
    print(results_df.head(15))
    print(f"\nFull ranked list saved to {output_path}")

if __name__ == '__main__':
    rank_candidates(
        model_path="output/exo_hunter_model.joblib",
        data_path=config.DATA_PATH
    )
