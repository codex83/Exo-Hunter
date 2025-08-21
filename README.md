# Exo-Hunter: A Machine Learning Pipeline for Exoplanet Discovery

A project to classify Kepler Objects of Interest (KOIs) as "Confirmed" exoplanets or "False Positives" using machine learning and light curve data.

The key finding is that the final **LightGBM model**, tuned with `RandomizedSearchCV`, **achieves over 99% accuracy**, demonstrating the power of feature engineering (specifically Signal-to-Noise Ratio) and gradient boosting on tabular astronomical data.

---

## Table of Contents
1. [Project Goal](#project-goal)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Project Structure](#project-structure)
6. [Getting Started](#getting-started)
7. [Usage](#usage)

---

## Project Goal

The objective was to build an end-to-end machine learning pipeline to automatically classify potential exoplanet candidates from the Kepler mission. This involved a systematic workflow including data preprocessing, feature engineering, model comparison, and hyperparameter tuning to produce a highly accurate and reliable classifier. The final model is then used to prioritize unclassified "Candidate" objects for further study.

---

## Dataset

*   **Source**: Kepler Objects of Interest (KOI) Data from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/data.html).
*   **Local Copy**: The full dataset is included in this repository at `data/cumulative.csv`.

---

## Methodology

### 1. Data Preprocessing (`src/preprocessing.py`)
*   **Data Separation**: The raw data was split into a primary dataset (for training and evaluation) containing 'CONFIRMED' and 'FALSE POSITIVE' labels, and a 'CANDIDATE' dataset for final prediction.
*   **Feature Selection**: Non-predictive or redundant columns (e.g., identifiers) were dropped based on the data dictionary.
*   **Imputation**: Missing numerical values were imputed using the median strategy to prevent data leakage and handle gaps in the data robustly.

### 2. Feature Engineering (`src/preprocessing.py`)
A crucial step was the creation of **Signal-to-Noise Ratio (SNR)** features for all physical measurements that had corresponding error values. For a given measurement `X` and its errors `X_err1`, `X_err2`, the SNR was calculated as `X / sqrt(X_err1^2 + X_err2^2)`. This proved to be a highly predictive feature transformation.

### 3. Model Evaluation (`src/model_evaluation.py`)
Three baseline models were compared using 5-fold stratified cross-validation to select the best architecture for this task: `RandomForestClassifier`, `MLPClassifier` (Neural Network), and `LGBMClassifier`. LightGBM was chosen for its superior performance in both accuracy and F1-score.

### 4. Hyperparameter Tuning & Final Training (`src/model_evaluation.py`, `main.py`)
The selected `LGBMClassifier` was tuned using `RandomizedSearchCV` over a wide parameter space to find the optimal hyperparameters. The best estimator was then re-trained on the entire training dataset to produce the final model, which was saved to `output/exo_hunter_model.joblib`.

---

## Results

The final, tuned model was evaluated on a held-out test set (20% of the primary data).

| Model              | Test Accuracy | Test F1-Score (Weighted) |
| ------------------ | ------------- | ------------------------ |
| **Tuned LightGBM** | **99.1%**     | **99.1%**                |

**Conclusion**: The systematic approach of feature engineering, robust validation, and hyperparameter tuning produced an exceptionally accurate model. The high performance indicates that the engineered SNR features and the inherent patterns in the Kepler data are highly separable with modern gradient boosting techniques.

### Performance Visualizations

| Confusion Matrix                                     | Feature Importance                                       |
| ---------------------------------------------------- | -------------------------------------------------------- |
| ![Confusion Matrix](plots/confusion_matrix.png)      | ![Feature Importance](plots/feature_importance.png)      |

The confusion matrix shows extremely low error rates, and the feature importance plot confirms that the engineered **SNR features** and existing **false positive flags** were the most influential predictors in the model.

### Prioritized Candidate List
The final model was used to predict the confirmation probability for 2,248 unclassified "Candidate" objects. The top 10 most promising candidates are listed below, providing a valuable resource for astronomers.

| kepoi_name | confirmation_probability |
|------------|--------------------------|
| K00315.01  | 1.0                      |
| K01475.02  | 1.0                      |
| K01871.02  | 1.0                      |
| K00865.01  | 1.0                      |
| K00415.01  | 1.0                      |
| K00211.01  | 1.0                      |
| K01466.01  | 1.0                      |
| K00339.03  | 1.0                      |
| K01239.02  | 1.0                      |
| K03319.01  | 1.0                      |

---

## Project Structure
```
.
├── data/
│   └── cumulative.csv
├── output/
│   ├── ranked_candidates.csv
│   └── exo_hunter_model.joblib
├── plots/
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── model_evaluation.py
│   └── preprocessing.py
├── .gitignore
├── main.py
├── rank_candidates.py
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites
* Python 3.8+
* An environment with the packages listed in `requirements.txt`.

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/codex83/Exo-Hunter.git
    cd Exo-Hunter
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
---

## Usage

The project is split into two primary scripts for training and prediction.

1.  **Run the Full Training Pipeline**:
    This script preprocesses the data, trains the model via hyperparameter tuning, evaluates it, and saves the final model artifact and plots.
    ```bash
    python main.py
    ```

2.  **Rank Candidates with the Trained Model**:
    This script loads the saved model from `output/` and uses it to predict on the candidate objects, saving the ranked list to `output/ranked_candidates.csv`.
    ```bash
    python rank_candidates.py
    ```
