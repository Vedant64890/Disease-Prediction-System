import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(train_path="csv_files/Training.csv", test_path="csv_files/Testing.csv"):
    """
    Load training and testing datasets.
    
    Args:
        train_path (str): Path to training CSV file
        test_path (str): Path to testing CSV file
    
    Returns:
        tuple: (train_data, test_data) as pandas DataFrames
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def clean_data(train_data, test_data):
    """
    Clean and standardize both datasets.
    
    Args:
        train_data (DataFrame): Raw training data
        test_data (DataFrame): Raw testing data
    
    Returns:
        tuple: (cleaned_train_data, cleaned_test_data)
    """
    # Remove duplicates
    train_data = train_data.drop_duplicates()
    test_data = test_data.drop_duplicates()
    
    # Remove unnamed columns
    train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
    test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]
    
    # Clean column names
    train_data.columns = train_data.columns.str.strip()
    test_data.columns = test_data.columns.str.strip()
    
    # Fill missing values
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    
    # Align columns between train and test
    train_features = [col for col in train_data.columns if col != "prognosis"]
    for col in train_features:
        if col not in test_data.columns:
            test_data[col] = 0
    test_data = test_data[[col for col in test_data.columns if col in train_features or col == "prognosis"]]
    
    return train_data, test_data


def preprocess_and_encode(train_data, test_data, test_size=0.2, random_state=42):
    """
    Preprocess data, separate features/targets, encode labels, and split train/validation.
    
    Args:
        train_data (DataFrame): Cleaned training data
        test_data (DataFrame): Cleaned testing data
        test_size (float): Validation split ratio
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Contains X_train_split, X_val, y_train_split, y_val, X_test, 
              y_test_encoded, label_encoder, and feature_names
    """
    print("\n--- PREPROCESSING & FEATURE ENGINEERING ---")
    
    # Separate features and target
    train_features = [col for col in train_data.columns if col != "prognosis"]
    X_train = train_data[train_features]
    y_train = train_data["prognosis"]
    
    X_test = test_data[train_features]
    y_test = test_data["prognosis"] if "prognosis" in test_data.columns else None
    
    print(f"\nFeatures shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    
    # Encode target variable (convert disease names to numbers)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    print(f"\nDisease Classes:")
    for i, disease in enumerate(label_encoder.classes_):
        print(f"{i}: {disease}")
    
    # Split training data into train and validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, 
        y_train_encoded, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_train_encoded
    )
    
    print(f"\nTraining split: {X_train_split.shape[0]} samples")
    print(f"Validation split: {X_val.shape[0]} samples")
    
    # Encode test data if available
    if y_test is not None:
        y_test_encoded = label_encoder.transform(y_test)
        print(f"Test set: {X_test.shape[0]} samples")
    else:
        y_test_encoded = None
        print(f"Test set: {X_test.shape[0]} samples (no target labels)")
    
    return {
        "X_train_split": X_train_split,
        "X_val": X_val,
        "y_train_split": y_train_split,
        "y_val": y_val,
        "X_test": X_test,
        "y_test_encoded": y_test_encoded,
        "label_encoder": label_encoder,
        "feature_names": train_features
    }


def save_preprocessing_data(data_dict, output_path="preprocessing_data.pkl"):
    """
    Save preprocessing results to a pickle file.
    
    Args:
        data_dict (dict): Dictionary with preprocessing results
        output_path (str): Path to output pickle file
    """
    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Preprocessed data saved to {output_path}!")


# ================================================================
# Main Execution
# ================================================================

if __name__ == "__main__":
    # Load data
    train_data, test_data = load_data()
    
    # Clean data
    train_data, test_data = clean_data(train_data, test_data)
    
    # Preprocess and encode
    preprocessing_data = preprocess_and_encode(train_data, test_data)
    
    # Save for next step
    save_preprocessing_data(preprocessing_data)
