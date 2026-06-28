import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def _predict_with_probabilities(model, feature_df, label_encoder):
    """
    Predict class and probabilities, with fallback for models lacking predict_proba.
    """
    prediction = np.asarray(model.predict(feature_df))[0]
    if np.issubdtype(np.asarray([prediction]).dtype, np.floating):
        prediction = int(np.rint(prediction))

    class_count = len(label_encoder.classes_)
    prediction = int(np.clip(prediction, 0, class_count - 1))

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_df)[0]
        class_labels = model.classes_
    else:
        probabilities = np.zeros(class_count, dtype=float)
        probabilities[prediction] = 1.0
        class_labels = np.arange(class_count)

    return prediction, probabilities, class_labels


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


def get_all_symptoms(model_metadata):
    """
    Get list of all available symptoms from training features.
    
    Args:
        model_metadata (dict): Model metadata
    
    Returns:
        list: List of all symptom names
    """
    return model_metadata["feature_names"]


def interactive_symptom_input(symptoms_list):
    """
    Interactive mode: Ask user for symptoms one by one.
    
    Args:
        symptoms_list (list): List of available symptoms
    
    Returns:
        dict: Dictionary with symptom selections
    """
    print("\n" + "="*70)
    print("DISEASE PREDICTION - INTERACTIVE MODE")
    print("="*70)
    
    print("\nAvailable Symptoms:")
    print("─"*70)
    for i, symptom in enumerate(symptoms_list, 1):
        print(f"{i:3d}. {symptom}")
    
    print("\n" + "─"*70)
    print("Enter symptom numbers (comma-separated) or 'all' to see descriptions:")
    print("Example: 1,5,12 (for symptoms 1, 5, and 12)")
    print("─"*70)
    
    selected_indices = input("\nSelect symptom numbers: ").strip()
    
    # Create symptom vector
    symptom_vector = {symptom: 0 for symptom in symptoms_list}
    
    if selected_indices.lower() == 'all':
        return None  # Signal for demo mode
    
    try:
        indices = [int(x.strip()) - 1 for x in selected_indices.split(',')]
        for idx in indices:
            if 0 <= idx < len(symptoms_list):
                symptom_vector[symptoms_list[idx]] = 1
            else:
                print(f"Warning: Index {idx+1} is out of range")
    except ValueError:
        print("Invalid input! Please enter numbers separated by commas.")
        return None
    
    return symptom_vector


def quick_symptom_input(symptoms_list):
    """
    Quick mode: Enter symptoms as comma-separated names.
    
    Args:
        symptoms_list (list): List of available symptoms
    
    Returns:
        dict: Dictionary with symptom selections
    """
    print("\n" + "="*70)
    print("DISEASE PREDICTION - QUICK MODE")
    print("="*70)
    
    print("\nEnter symptoms (comma-separated).")
    print("Available symptoms: " + ", ".join(symptoms_list[:5]) + "...")
    
    user_input = input("\nSymptoms: ").strip()
    
    symptom_vector = {symptom: 0 for symptom in symptoms_list}
    
    user_symptoms = [s.strip().lower() for s in user_input.split(',')]
    
    for user_symptom in user_symptoms:
        for available_symptom in symptoms_list:
            if user_symptom in available_symptom.lower():
                symptom_vector[available_symptom] = 1
                break
    
    return symptom_vector


def demo_prediction(model, model_metadata):
    """
    Run prediction with demo symptoms.
    
    Args:
        model: Trained model
        model_metadata (dict): Model metadata
    """
    print("\n" + "="*70)
    print("DISEASE PREDICTION - DEMO MODE")
    print("="*70)
    
    symptoms_list = model_metadata["feature_names"]
    label_encoder = model_metadata["label_encoder"]
    
    # Demo case 1: Flu-like symptoms
    print("\n" + "─"*70)
    print("DEMO CASE 1: Flu-like Symptoms")
    print("─"*70)
    
    demo1_symptoms = {s: 0 for s in symptoms_list}
    # Set some common cold symptoms
    for symptom in symptoms_list:
        if any(word in symptom.lower() for word in ['cough', 'sneezing', 'runny_nose', 'fatigue', 'headache']):
            demo1_symptoms[symptom] = 1
    
    # Make prediction with DataFrame
    feature_data = [demo1_symptoms[s] for s in symptoms_list]
    feature_df = pd.DataFrame([feature_data], columns=symptoms_list)
    prediction, probabilities, _ = _predict_with_probabilities(model, feature_df, label_encoder)
    
    print(f"\nSymptoms entered: {sum(demo1_symptoms.values())} symptoms")
    print(f"\nPredicted Disease: {label_encoder.inverse_transform([prediction])[0]}")
    
    # Top 5 predictions
    top_5_idx = np.argsort(probabilities)[-5:][::-1]
    print(f"\nTop 5 Predictions:")
    for rank, idx in enumerate(top_5_idx, 1):
        disease = label_encoder.inverse_transform([idx])[0]
        confidence = probabilities[idx] * 100
        print(f"{rank}. {disease}: {confidence:.2f}%")
    
    # Demo case 2: Skin symptoms
    print("\n" + "─"*70)
    print("DEMO CASE 2: Skin Symptoms")
    print("─"*70)
    
    demo2_symptoms = {s: 0 for s in symptoms_list}
    for symptom in symptoms_list:
        if any(word in symptom.lower() for word in ['rash', 'itching', 'skin_lesions', 'pustules']):
            demo2_symptoms[symptom] = 1
    
    feature_data = [demo2_symptoms[s] for s in symptoms_list]
    feature_df = pd.DataFrame([feature_data], columns=symptoms_list)
    prediction, probabilities, _ = _predict_with_probabilities(model, feature_df, label_encoder)
    
    print(f"\nSymptoms entered: {sum(demo2_symptoms.values())} symptoms")
    print(f"\nPredicted Disease: {label_encoder.inverse_transform([prediction])[0]}")
    
    top_5_idx = np.argsort(probabilities)[-5:][::-1]
    print(f"\nTop 5 Predictions:")
    for rank, idx in enumerate(top_5_idx, 1):
        disease = label_encoder.inverse_transform([idx])[0]
        confidence = probabilities[idx] * 100
        print(f"{rank}. {disease}: {confidence:.2f}%")


def make_prediction(symptom_vector, model, model_metadata):

    symptoms_list = model_metadata["feature_names"]
    label_encoder = model_metadata["label_encoder"]
    
    # Create feature DataFrame (with proper feature names)
    feature_data = [symptom_vector[s] for s in symptoms_list]
    feature_df = pd.DataFrame([feature_data], columns=symptoms_list)
    
    # Make prediction
    prediction, probabilities, class_labels = _predict_with_probabilities(model, feature_df, label_encoder)
    
    predicted_disease = label_encoder.inverse_transform([prediction])[0]
    predicted_class_pos = np.where(class_labels == prediction)[0][0]
    confidence = probabilities[predicted_class_pos] * 100
    
    # Get top 5 predictions
    top_5_idx = np.argsort(probabilities)[-5:][::-1]
    top_5_predictions = []
    for idx in top_5_idx:
        class_value = class_labels[idx]
        disease = label_encoder.inverse_transform([class_value])[0]
        confidence_pct = probabilities[idx] * 100
        top_5_predictions.append((disease, confidence_pct))
    
    return predicted_disease, top_5_predictions, confidence


def display_prediction_results(predicted_disease, top_5_predictions, confidence, symptom_count):
    """
    Display prediction results in a nice format.
    
    Args:
        predicted_disease (str): Predicted disease
        top_5_predictions (list): Top 5 predictions with confidence
        confidence (float): Confidence of main prediction
        symptom_count (int): Number of symptoms provided
    """
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    print(f"\nSymptoms Provided: {symptom_count}")
    print(f"\n{'PRIMARY PREDICTION':^70}")
    print("─"*70)
    print(f"Disease: {predicted_disease}")
    print(f"Confidence: {confidence:.2f}%")
    
    print(f"\n{'TOP 5 PREDICTIONS':^70}")
    print("─"*70)
    for rank, (disease, conf) in enumerate(top_5_predictions, 1):
        bar_length = int(conf / 2)  # Scale to 50 chars max
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{rank}. {disease:40s} {bar} {conf:.2f}%")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DISEASE PREDICTION SYSTEM")
    print("="*70)
    
    # Load model
    print("\nLoading trained model...")
    model_metadata = load_best_model()
    model = model_metadata["model"]
    print(f"✓ Model loaded: {model_metadata['model_name']}")
    print(f"✓ Accuracy: {model_metadata['accuracy']:.4f}")
    
    symptoms_list = get_all_symptoms(model_metadata)
    print(f"✓ Total symptoms: {len(symptoms_list)}")
    
    # Menu
    while True:
        print("\n" + "="*70)
        print("MAIN MENU")
        print("="*70)
        print("1. Demo Mode (See example predictions)")
        print("2. Interactive Mode (Select symptoms by number)")
        print("3. Quick Mode (Enter symptom names)")
        print("4. View All Symptoms")
        print("5. Exit")
        print("="*70)
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            demo_prediction(model, model_metadata)
        
        elif choice == "2":
            symptom_vector = interactive_symptom_input(symptoms_list)
            if symptom_vector:
                symptom_count = sum(symptom_vector.values())
                if symptom_count > 0:
                    predicted_disease, top_5, confidence = make_prediction(
                        symptom_vector, model, model_metadata
                    )
                    display_prediction_results(predicted_disease, top_5, confidence, symptom_count)
                else:
                    print("\n⚠ No symptoms selected. Please try again.")
        
        elif choice == "3":
            symptom_vector = quick_symptom_input(symptoms_list)
            if symptom_vector:
                symptom_count = sum(symptom_vector.values())
                if symptom_count > 0:
                    predicted_disease, top_5, confidence = make_prediction(
                        symptom_vector, model, model_metadata
                    )
                    display_prediction_results(predicted_disease, top_5, confidence, symptom_count)
                else:
                    print("\n⚠ No matching symptoms found. Please check symptom names.")
        
        elif choice == "4":
            print("\n" + "="*70)
            print("ALL AVAILABLE SYMPTOMS")
            print("="*70)
            for i, symptom in enumerate(symptoms_list, 1):
                print(f"{i:3d}. {symptom}")
        
        elif choice == "5":
            print("\nThank you for using Disease Prediction System!")
            break
        
        else:
            print("\n⚠ Invalid option. Please select 1-5.")
