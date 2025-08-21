# main.py
from src import config
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model_evaluation import tune_and_train_best_model
from src.evaluation import evaluate_model, plot_feature_importance
from sklearn.model_selection import train_test_split

def main():
    """Main function to run the Exo-Hunter project workflow."""
    
    # --- Phase 1 & 2: Data Loading and Preprocessing ---
    raw_df = load_data(config.DATA_PATH)
    if raw_df is None:
        return
        
    primary_df, candidates_df = preprocess_data(raw_df, config.COLS_TO_DROP, config.TARGET_COLUMN)
    
    # --- Split Data for Training and Final Evaluation ---
    X = primary_df.drop(config.TARGET_COLUMN, axis=1)
    y = primary_df[config.TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    # --- Phase 4: Hyperparameter Tuning and Final Model Training ---
    best_model_pipeline = tune_and_train_best_model(X_train, y_train, config.RANDOM_STATE)
    
    print("\nFinal model pipeline has been trained.")
    
    # --- Phase 5: Final Evaluation ---
    evaluate_model(best_model_pipeline, X_test, y_test)
    plot_feature_importance(best_model_pipeline, X_train.columns)

if __name__ == '__main__':
    main()
