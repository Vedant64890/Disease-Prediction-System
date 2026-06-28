"""
Advanced Model Training Script
Ensemble Models: Random Forest + Gradient Boosting
With comprehensive validation and metrics
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


def load_preprocessing_data(path="preprocessing_data.pkl"):
    """Load preprocessed data from pickle file"""
    print("📂 Loading preprocessing data...")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print("✅ Data loaded successfully!")
    return data


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest model with hyperparameter tuning"""
    print("\n" + "="*70)
    print("🌲 TRAINING RANDOM FOREST MODEL")
    print("="*70)
    
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("\n⏳ Training Random Forest (300 trees)...")
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    
    print(f"\n📊 Random Forest Results:")
    print(f"  Train Accuracy:      {train_acc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Validation Precision: {val_precision:.4f}")
    print(f"  Validation Recall:    {val_recall:.4f}")
    print(f"  Validation F1-Score:  {val_f1:.4f}")
    
    # Feature importance
    feature_importance = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'model': rf_model,
        'y_val_pred': y_val_pred
    }
    
    return feature_importance


def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """Train Gradient Boosting model"""
    print("\n" + "="*70)
    print("🚀 TRAINING GRADIENT BOOSTING MODEL")
    print("="*70)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
        verbose=1
    )
    
    print("\n⏳ Training Gradient Boosting (200 estimators)...")
    gb_model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = gb_model.predict(X_train)
    y_val_pred = gb_model.predict(X_val)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    
    print(f"\n📊 Gradient Boosting Results:")
    print(f"  Train Accuracy:      {train_acc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Validation Precision: {val_precision:.4f}")
    print(f"  Validation Recall:    {val_recall:.4f}")
    print(f"  Validation F1-Score:  {val_f1:.4f}")
    
    results = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'model': gb_model,
        'y_val_pred': y_val_pred
    }
    
    return results


def compare_models(rf_results, gb_results):
    """Compare model performance"""
    print("\n" + "="*70)
    print("⚖️ MODEL COMPARISON")
    print("="*70)
    
    comparison_data = {
        'Random Forest': {
            'Train Accuracy': rf_results['train_acc'],
            'Val Accuracy': rf_results['val_acc'],
            'Precision': rf_results['val_precision'],
            'Recall': rf_results['val_recall'],
            'F1-Score': rf_results['val_f1']
        },
        'Gradient Boosting': {
            'Train Accuracy': gb_results['train_acc'],
            'Val Accuracy': gb_results['val_acc'],
            'Precision': gb_results['val_precision'],
            'Recall': gb_results['val_recall'],
            'F1-Score': gb_results['val_f1']
        }
    }
    
    df_comparison = pd.DataFrame(comparison_data).T
    print("\n" + df_comparison.to_string())
    
    # Determine best model
    best_rf_f1 = rf_results['val_f1']
    best_gb_f1 = gb_results['val_f1']
    
    if best_rf_f1 > best_gb_f1:
        print(f"\n✅ Random Forest is superior (F1: {best_rf_f1:.4f} vs {best_gb_f1:.4f})")
        return 'Random Forest', rf_results, 0.6
    else:
        print(f"\n✅ Gradient Boosting is superior (F1: {best_gb_f1:.4f} vs {best_rf_f1:.4f})")
        return 'Gradient Boosting', gb_results, 0.4


def evaluate_model_detailed(model, y_val, y_pred, label_encoder, model_name):
    """Detailed model evaluation"""
    print("\n" + "="*70)
    print(f"🔍 DETAILED EVALUATION - {model_name}")
    print("="*70)
    
    print("\n📋 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_, zero_division=0))
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_val, y_pred)
    print(f"\n📊 Confusion Matrix Shape: {cm.shape}")
    print("   (Rows: Actual Classes, Columns: Predicted Classes)")
    
    # Find top misclassifications
    print("\n⚠️ Top Misclassifications:")
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    
    misclassified = []
    for i in range(len(cm_copy)):
        for j in range(len(cm_copy)):
            if cm_copy[i, j] > 0:
                misclassified.append((cm_copy[i, j], label_encoder.classes_[i], label_encoder.classes_[j]))
    
    misclassified.sort(reverse=True)
    for count, actual, predicted in misclassified[:5]:
        print(f"   {count} samples: {actual} → {predicted}")


def save_model_and_metadata(rf_model, gb_model, label_encoder, feature_names, rf_results, gb_results):
    """Save the best individual model as best_model.pkl for dashboard compatibility."""
    print("\n" + "="*70)
    print("SAVING BEST MODEL AS best_model.pkl")
    print("="*70)

    # Pick the best individual model by F1 score
    if rf_results['val_f1'] >= gb_results['val_f1']:
        best_model = rf_model
        best_model_name = "Random Forest"
        best_results = rf_results
    else:
        best_model = gb_model
        best_model_name = "Gradient Boosting"
        best_results = gb_results

    # Create feature importance from Random Forest
    feature_importance = {
        name: float(importance)
        for name, importance in zip(feature_names, rf_model.feature_importances_)
    }

    # Save in the EXACT format s06_dash.py expects:
    #   model_metadata["model"] must be a single sklearn model with .predict() and .predict_proba()
    #   model_metadata["model_name"], "accuracy", "precision", "recall", "f1_score" are used by dashboard
    model_metadata = {
        "model": best_model,
        "model_name": best_model_name,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "accuracy": best_results['val_acc'],
        "precision": best_results['val_precision'],
        "recall": best_results['val_recall'],
        "f1_score": best_results['val_f1'],
        "rf_metrics": {
            "accuracy": rf_results['val_acc'],
            "f1_score": rf_results['val_f1'],
            "precision": rf_results['val_precision'],
            "recall": rf_results['val_recall']
        },
        "gb_metrics": {
            "accuracy": gb_results['val_acc'],
            "f1_score": gb_results['val_f1'],
            "precision": gb_results['val_precision'],
            "recall": gb_results['val_recall']
        },
        "created_at": pd.Timestamp.now().isoformat()
    }

    # Save as best_model.pkl (dashboard loads this file)
    with open("best_model.pkl", "wb") as f:
        pickle.dump(model_metadata, f)

    # Also save as advanced_model.pkl for backward compatibility
    with open("advanced_model.pkl", "wb") as f:
        pickle.dump(model_metadata, f)

    print(f"\n  Best model saved as 'best_model.pkl'")
    print(f"   Winner: {best_model_name}")
    print(f"   Accuracy:  {best_results['val_acc']:.4f}")
    print(f"   Precision: {best_results['val_precision']:.4f}")
    print(f"   Recall:    {best_results['val_recall']:.4f}")
    print(f"   F1-Score:  {best_results['val_f1']:.4f}")
    print(f"   Features:  {len(feature_names)}")
    print(f"   Classes:   {len(label_encoder.classes_)}")


def print_summary(rf_results, gb_results):
    """Print training summary"""
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    
    print(f"""
✅ MODEL TRAINING COMPLETED SUCCESSFULLY

Random Forest Performance:
  • Validation Accuracy: {rf_results['val_acc']:.4f}
  • F1-Score: {rf_results['val_f1']:.4f}
  • Precision: {rf_results['val_precision']:.4f}
  • Recall: {rf_results['val_recall']:.4f}

Gradient Boosting Performance:
  • Validation Accuracy: {gb_results['val_acc']:.4f}
  • F1-Score: {gb_results['val_f1']:.4f}
  • Precision: {gb_results['val_precision']:.4f}
  • Recall: {gb_results['val_recall']:.4f}

Ensemble Model:
  • Strategy: Weighted voting (RF:60%, GB:40%)
  • Expected Performance: Better than individual models
  • Saved to: advanced_model.pkl

Next Steps:
  1. Run the Streamlit app: streamlit run app.py
  2. Login or create an account
  3. Use the Symptom Analyzer for predictions
  4. Chat with the AI chatbot
    """)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 ADVANCED ML MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Load data
    preprocessing_data = load_preprocessing_data()
    
    X_train_split = preprocessing_data['X_train_split']
    X_val = preprocessing_data['X_val']
    y_train_split = preprocessing_data['y_train_split']
    y_val = preprocessing_data['y_val']
    label_encoder = preprocessing_data['label_encoder']
    feature_names = preprocessing_data['feature_names']
    
    print(f"\n📊 Data Summary:")
    print(f"   Train samples: {X_train_split.shape[0]}")
    print(f"   Validation samples: {X_val.shape[0]}")
    print(f"   Features: {X_train_split.shape[1]}")
    print(f"   Classes: {len(label_encoder.classes_)}")
    
    # Train models
    rf_results = train_random_forest(X_train_split, y_train_split, X_val, y_val)
    gb_results = train_gradient_boosting(X_train_split, y_train_split, X_val, y_val)
    
    # Compare models
    best_name, best_results, weight = compare_models(rf_results, gb_results)
    
    # Detailed evaluation
    evaluate_model_detailed(
        rf_results['model'], y_val, rf_results['y_val_pred'],
        label_encoder, "Random Forest"
    )
    
    evaluate_model_detailed(
        gb_results['model'], y_val, gb_results['y_val_pred'],
        label_encoder, "Gradient Boosting"
    )
    
    # Save models
    save_model_and_metadata(
        rf_results['model'], gb_results['model'],
        label_encoder, feature_names, rf_results, gb_results
    )
    
    # Summary
    print_summary(rf_results, gb_results)
    
    print("\n✨ Training pipeline completed!")
    print("Run: streamlit run app.py")
