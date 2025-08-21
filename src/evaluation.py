# src/evaluation.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model_pipeline, X_test, y_test):
    """Evaluates the final model on the test set and prints reports."""
    print("\n--- Phase 5: Final Model Evaluation on Test Set ---")
    
    y_pred = model_pipeline.predict(X_test)
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['FALSE POSITIVE', 'CONFIRMED'])
    print(report)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FALSE POSITIVE', 'CONFIRMED'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def plot_feature_importance(model_pipeline, feature_names):
    """Extracts and plots feature importances from the trained model."""
    print("\n--- Feature Importance Analysis ---")
    
    classifier = model_pipeline.named_steps['classifier']
    importances = classifier.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    print("Top 15 most important features:")
    print(feature_importance_df.head(15))
    
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='importance',
        y='feature',
        data=feature_importance_df.head(15),
        palette='viridis'
    )
    plt.title('Top 15 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
