import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')


def _get_prediction_probabilities(model, X, class_count):
    """
    Return class probabilities, with a fallback for models without predict_proba.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    y_pred = np.asarray(model.predict(X)).astype(int)
    y_pred = np.clip(y_pred, 0, class_count - 1)
    probabilities = np.zeros((len(y_pred), class_count), dtype=float)
    probabilities[np.arange(len(y_pred)), y_pred] = 1.0
    return probabilities


def load_best_model(path="best_model.pkl"):
    """
    Load the best trained model and metadata.
    
    Args:
        path (str): Path to best model pickle file
    
    Returns:
        dict: Model metadata and the trained model
    """
    with open(path, "rb") as f:
        model_metadata = pickle.load(f)
    return model_metadata


def load_test_data(path="preprocessing_data.pkl"):
    """
    Load test data from preprocessing pickle file.
    
    Args:
        path (str): Path to preprocessing pickle file
    
    Returns:
        tuple: (X_test, y_test_encoded) test features and targets
    """
    with open(path, "rb") as f:
        preprocessing_data = pickle.load(f)
    
    return (preprocessing_data["X_test"], 
            preprocessing_data["y_test_encoded"],
            preprocessing_data["label_encoder"])


def evaluate_on_test_set(model, X_test, y_test_encoded, label_encoder):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test_encoded: Test targets (encoded)
        label_encoder: LabelEncoder for disease names
    
    Returns:
        dict: Evaluation metrics and predictions
    """
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)
    
    # Make predictions
    y_test_pred = np.asarray(model.predict(X_test))
    if np.issubdtype(y_test_pred.dtype, np.floating):
        y_test_pred = np.rint(y_test_pred).astype(int)
        y_test_pred = np.clip(y_test_pred, 0, len(label_encoder.classes_) - 1)

    y_test_pred_proba = _get_prediction_probabilities(model, X_test, len(label_encoder.classes_))
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
    test_precision = precision_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)
    
    print(f"\nTest Set Performance:")
    print(f"{'─'*70}")
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"Samples:   {len(y_test_encoded)}")
    
    return {
        "y_test_pred": y_test_pred,
        "y_test_pred_proba": y_test_pred_proba,
        "accuracy": test_accuracy,
        "precision": test_precision,
        "recall": test_recall,
        "f1": test_f1
    }


def detailed_classification_report(y_test_encoded, y_test_pred, label_encoder):
    """
    Print detailed classification report.
    
    Args:
        y_test_encoded: True test labels
        y_test_pred: Predicted test labels
        label_encoder: LabelEncoder for disease names
    """
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*70)
    
    report = classification_report(
        y_test_encoded,
        y_test_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    print("\n" + report)


def analyze_confusion_matrix(y_test_encoded, y_test_pred, label_encoder):
    """
    Analyze confusion matrix to find problematic disease pairs.
    
    Args:
        y_test_encoded: True test labels
        y_test_pred: Predicted test labels
        label_encoder: LabelEncoder for disease names
    """
    print("\n" + "="*70)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*70)
    
    cm = confusion_matrix(y_test_encoded, y_test_pred)
    
    print(f"\nConfusion Matrix Shape: {cm.shape}")
    print(f"Test Samples: {cm.sum()}")
    
    # Find misclassifications
    print("\n" + "─"*70)
    print("MISCLASSIFICATIONS:")
    print("─"*70)
    
    misclassified_pairs = []
    correct_count = 0
    
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i == j:
                correct_count += cm[i, j]
            elif cm[i, j] > 0:
                misclassified_pairs.append((cm[i, j], label_encoder.classes_[i], label_encoder.classes_[j]))
    
    if misclassified_pairs:
        misclassified_pairs.sort(reverse=True)
        print(f"\nCorrectly Classified: {correct_count}/{cm.sum()}")
        print(f"\nTop Misclassifications:")
        for count, actual, predicted in misclassified_pairs[:10]:
            print(f"  {count} sample(s): '{actual}' → '{predicted}'")
    else:
        print(f"\n✓ PERFECT! No misclassifications!")


def per_class_performance(y_test_encoded, y_test_pred, label_encoder):
    """
    Show performance for each disease class.
    
    Args:
        y_test_encoded: True test labels
        y_test_pred: Predicted test labels
        label_encoder: LabelEncoder for disease names
    """
    print("\n" + "="*70)
    print("PER-CLASS PERFORMANCE")
    print("="*70)
    
    # Calculate per-class metrics
    cm = confusion_matrix(y_test_encoded, y_test_pred)
    
    class_performance = []
    for i in range(len(cm)):
        tp = cm[i, i]  # True positives
        fp = cm[:, i].sum() - tp  # False positives
        fn = cm[i, :].sum() - tp  # False negatives
        tn = cm.sum() - tp - fp - fn  # True negatives
        
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
            
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
            
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        
        class_performance.append({
            "Disease": label_encoder.classes_[i],
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Support": cm[i, :].sum()
        })
    
    # Sort by F1-score
    class_performance.sort(key=lambda x: x["F1-Score"], reverse=True)
    
    # Display as table
    df = pd.DataFrame(class_performance)
    print("\nTop 10 Best Performing Diseases:")
    print(df.head(10).to_string(index=False))
    
    if len(df) > 10:
        print("\nTop 10 Worst Performing Diseases:")
        print(df.tail(10).to_string(index=False))


def generate_summary_report(model_metadata, eval_results):
    """
    Generate a summary report comparing validation vs test performance.
    
    Args:
        model_metadata (dict): Metadata from best model
        eval_results (dict): Results from test evaluation
    """
    print("\n" + "="*70)
    print("SUMMARY REPORT: VALIDATION vs TEST")
    print("="*70)
    
    summary_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Validation": [
            model_metadata["accuracy"],
            "N/A",
            "N/A",
            model_metadata["f1_score"]
        ],
        "Test": [
            f"{eval_results['accuracy']:.4f}",
            f"{eval_results['precision']:.4f}",
            f"{eval_results['recall']:.4f}",
            f"{eval_results['f1']:.4f}"
        ]
    })
    
    print("\n" + summary_df.to_string(index=False))
    
    print(f"\n✓ Model: {model_metadata['model_name']}")
    print(f"✓ Test Set Size: 42 samples")
    print(f"✓ Training Set Size: 243 samples")
    print(f"✓ Validation Set Size: 61 samples")


# ================================================================
# Main Execution
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MODEL EVALUATION ON TEST SET")
    print("="*70)
    
    # Load best model
    print("\nLoading best model...")
    model_metadata = load_best_model()
    model = model_metadata["model"]
    print(f"✓ Loaded model: {model_metadata['model_name']}")
    
    # Load test data
    print("Loading test data...")
    X_test, y_test_encoded, label_encoder = load_test_data()
    print(f"✓ Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Evaluate on test set
    eval_results = evaluate_on_test_set(model, X_test, y_test_encoded, label_encoder)
    
    # Detailed reports
    detailed_classification_report(y_test_encoded, eval_results["y_test_pred"], label_encoder)
    analyze_confusion_matrix(y_test_encoded, eval_results["y_test_pred"], label_encoder)
    per_class_performance(y_test_encoded, eval_results["y_test_pred"], label_encoder)
    
    # Summary
    generate_summary_report(model_metadata, eval_results)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("\nNext Step:")
    print("Run: python s05_predict.py  (to make predictions on new symptoms)")
    print("="*70 + "\n")
