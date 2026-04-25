import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

def _to_class_predictions(y_pred, y_reference):
    """
    Convert model outputs to valid class labels when needed (e.g., Linear Regression).
    """
    y_pred_array = np.asarray(y_pred)
    if np.issubdtype(y_pred_array.dtype, np.floating):
        class_min = int(np.min(y_reference))
        class_max = int(np.max(y_reference))
        y_pred_array = np.rint(y_pred_array).astype(int)
        y_pred_array = np.clip(y_pred_array, class_min, class_max)
    return y_pred_array


def load_preprocessing_data(path="preprocessing_data.pkl"):
    """
    Load preprocessed data from pickle file.
    
    Args:
        path (str): Path to preprocessing pickle file
    
    Returns:
        dict: Contains preprocessed data and metadata
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def train_models(X_train, y_train, X_val, y_val):
    """
    Train multiple ML models and evaluate on validation set.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
    
    Returns:
        dict: Models with their training and validation metrics
    """
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Linear Regression": LinearRegression(n_jobs=-1)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'─'*70}")
        print(f"Training: {model_name}")
        print(f"{'─'*70}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict on train and validation
        y_train_pred = _to_class_predictions(model.predict(X_train), y_train)
        y_val_pred = _to_class_predictions(model.predict(X_val), y_train)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
        
        results[model_name] = {
            "model": model,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "y_val_pred": y_val_pred
        }
        
        print(f"Train Accuracy:     {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation Precision: {val_precision:.4f}")
        print(f"Validation Recall:   {val_recall:.4f}")
        print(f"Validation F1-Score: {val_f1:.4f}")
    
    return results


def compare_models(results):
    """
    Compare all models and identify the best one.
    
    Args:
        results (dict): Model results from train_models()
    
    Returns:
        tuple: (best_model_name, best_results)
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        model_name: {
            "Train Accuracy": result["train_accuracy"],
            "Val Accuracy": result["val_accuracy"],
            "Val Precision": result["val_precision"],
            "Val Recall": result["val_recall"],
            "Val F1-Score": result["val_f1"]
        }
        for model_name, result in results.items()
    }).T
    
    print("\n" + comparison_df.to_string())
    
    # Find best model by validation accuracy
    best_model_name = max(results.keys(), key=lambda x: results[x]["val_accuracy"])
    best_val_f1 = max(results.keys(), key=lambda x: results[x]["val_f1"])
    
    print(f"\n✓ Best Model (by Accuracy): {best_model_name}")
    print(f"✓ Best Model (by F1-Score): {best_val_f1}")
    
    return best_model_name, results[best_model_name]


def evaluate_best_model(best_model_name, best_results, y_val, label_encoder):
    """
    Detailed evaluation of best model including classification report and confusion matrix.
    
    Args:
        best_model_name (str): Name of best model
        best_results (dict): Results dictionary for best model
        y_val: Validation targets
        label_encoder: LabelEncoder for disease names
    """
    print("\n" + "="*70)
    print(f"DETAILED EVALUATION - {best_model_name}")
    print("="*70)
    
    y_val_pred = best_results["y_val_pred"]
    
    print("\nClassification Report:")
    print(classification_report(
        y_val, 
        y_val_pred, 
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    print(f"\nConfusion Matrix Shape: {cm.shape}")
    print("(Rows: Actual, Columns: Predicted)")
    
    # Find most confused classes
    print("\nTop Misclassifications:")
    np.fill_diagonal(cm, 0)  # Zero out correct predictions
    misclassified_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j] > 0:
                misclassified_pairs.append((cm[i, j], label_encoder.classes_[i], label_encoder.classes_[j]))
    
    misclassified_pairs.sort(reverse=True)
    for count, actual, predicted in misclassified_pairs[:5]:
        print(f"  {count} samples: {actual} → {predicted}")


def save_best_model(best_model_name, best_results, feature_names, label_encoder):
    """
    Save best model and necessary metadata for future predictions.
    
    Args:
        best_model_name (str): Name of best model
        best_results (dict): Results dictionary for best model
        feature_names (list): List of feature names
        label_encoder: LabelEncoder for disease names
    """
    model_metadata = {
        "model": best_results["model"],
        "model_name": best_model_name,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
        "accuracy": best_results["val_accuracy"],
        "f1_score": best_results["val_f1"]
    }
    
    # Save model
    with open("best_model.pkl", "wb") as f:
        pickle.dump(model_metadata, f)
    
    print(f"\n✓ Best model saved as 'best_model.pkl'")
    print(f"  Model: {best_model_name}")
    print(f"  Accuracy: {best_results['val_accuracy']:.4f}")
    print(f"  F1-Score: {best_results['val_f1']:.4f}")




if __name__ == "__main__":
    print("\n" + "="*70)
    print("DISEASE PREDICTION MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    preprocessing_data = load_preprocessing_data()
    
    X_train_split = preprocessing_data["X_train_split"]
    X_val = preprocessing_data["X_val"]
    y_train_split = preprocessing_data["y_train_split"]
    y_val = preprocessing_data["y_val"]
    X_test = preprocessing_data["X_test"]
    label_encoder = preprocessing_data["label_encoder"]
    feature_names = preprocessing_data["feature_names"]
    
    print(f" Data loaded successfully")
    print(f"  Training samples: {X_train_split.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train_split.shape[1]}")
    print(f"  Disease classes: {len(label_encoder.classes_)}")
    
    # Train multiple models
    results = train_models(X_train_split, y_train_split, X_val, y_val)
    
    # Compare models
    best_model_name, best_results = compare_models(results)
    
    # Detailed evaluation
    evaluate_best_model(best_model_name, best_results, y_val, label_encoder)
    
    # Save best model
    save_best_model(best_model_name, best_results, feature_names, label_encoder)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Run: python s04_eval.py  (to evaluate on test set)")
    print("2. Run: python s05_predict.py  (to make predictions on new data)")
    print("="*70 + "\n")
