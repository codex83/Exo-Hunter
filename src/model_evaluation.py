# src/model_evaluation.py
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import randint, uniform

def evaluate_baseline_models(X_train_processed, y_train, random_state: int):
    """Evaluates baseline models using cross-validation."""
    print("\n--- Comparing Models using 5-Fold Cross-Validation ---")
    models = {
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'LightGBM': LGBMClassifier(random_state=random_state, verbose=-1),
        'Neural Network (MLP)': MLPClassifier(random_state=random_state, max_iter=1000)
    }
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        accuracy_scores = cross_val_score(model, X_train_processed, y_train, cv=cv_strategy, scoring='accuracy')
        print(f"  Mean Accuracy: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std():.4f})")
        f1_scores = cross_val_score(model, X_train_processed, y_train, cv=cv_strategy, scoring='f1_weighted')
        print(f"  Mean F1-Score: {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")

def tune_and_train_best_model(X_train, y_train, random_state: int):
    """Tunes hyperparameters for LightGBM and trains the final model."""
    print("\n--- Phase 4: Hyperparameter Tuning for LightGBM ---")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LGBMClassifier(random_state=random_state, verbose=-1))
    ])
    param_dist = {
        'classifier__n_estimators': randint(100, 1000),
        'classifier__learning_rate': uniform(0.01, 0.3),
        'classifier__num_leaves': randint(20, 100),
        'classifier__max_depth': randint(-1, 50),
        'classifier__subsample': uniform(0.6, 0.4),
        'classifier__colsample_bytree': uniform(0.6, 0.4)
    }
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=50,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        scoring='f1_weighted',
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    print("Starting hyperparameter tuning...")
    random_search.fit(X_train, y_train)
    print(f"\nBest F1-Score from search: {random_search.best_score_:.4f}")
    print("Best parameters found:")
    print(random_search.best_params_)
    return random_search.best_estimator_
