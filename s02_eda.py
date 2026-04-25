from s01_prep import load_data, clean_data

# ================================================================
# LOAD AND EXPLORE DATASETS
# ================================================================

# Load datasets
train_data, test_data = load_data()

# Display training data info
print("\n--- TRAINING DATA (RAW) ---")
print(f"Rows: {train_data.shape[0]}")
print(f"Columns: {train_data.shape[1]}")
print(f"Column names: {list(train_data.columns)}")

# Display testing data info
print("\n--- TESTING DATA (RAW) ---")
print(f"Rows: {test_data.shape[0]}")
print(f"Columns: {test_data.shape[1]}")
print(f"Column names: {list(test_data.columns)}")

# Check target variable distribution
print("\n--- TARGET VARIABLE ---")
if "prognosis" in train_data.columns:
    disease_count = train_data["prognosis"].value_counts()
    print(f"Number of diseases: {len(disease_count)}")
    print(f"\nDisease distribution:")
    print(disease_count)

# Check for missing values
print("\n--- DATA QUALITY ---")
train_missing = train_data.isnull().sum().sum()
test_missing = test_data.isnull().sum().sum()
print(f"Training data values: {train_missing}")
print(f"Testing data missing values: {test_missing}")

# Check for duplicate rows
train_duplicates = train_data.duplicated().sum()
test_duplicates = test_data.duplicated().sum()
print(f"Duplicate rows in training: {train_duplicates}")
print(f"Duplicate rows in testing: {test_duplicates}")

# Store before cleaning counts
train_rows_before = train_data.shape[0]
train_cols_before = train_data.shape[1]
test_rows_before = test_data.shape[0]
test_cols_before = test_data.shape[1]

print("\n--- BEFORE CLEANING ---")
print(f"Training: {train_rows_before} rows, {train_cols_before} columns")
print(f"Testing: {test_rows_before} rows, {test_cols_before} columns")

# ================================================================
# DATA CLEANING (Using preprocessing module)
# ================================================================

print("\n\n--- CLEANING DATA ---")
train_data, test_data = clean_data(train_data, test_data)
print(f"Removed duplicates")
print(f"Removed unnamed columns")
print(f"Trimmed column names")
print(f"Filled missing values with 0")
print(f"Aligned test data columns to match training data")

# Store after cleaning counts
train_rows_after = train_data.shape[0]
train_cols_after = train_data.shape[1]
test_rows_after = test_data.shape[0]
test_cols_after = test_data.shape[1]

print(f"\n--- AFTER CLEANING ---")
print(f"Training: {train_rows_after} rows, {train_cols_after} columns")
print(f"Testing: {test_rows_after} rows, {test_cols_after} columns")

# ================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ================================================================

print("\n\n--- EXPLORATORY DATA ANALYSIS ---")

# Target variable distribution
print("\nTarget Variable Distribution (Disease Classes):")
target_dist = train_data["prognosis"].value_counts()
print(target_dist)

# Basic statistics
print("\nBasic Statistics of Features:")
feature_columns = [col for col in train_data.columns if col != "prognosis"]
print(train_data[feature_columns].describe())

# Check class balance
print("\nClass Balance:")
total_samples = len(train_data)
for disease, count in target_dist.items():
    percentage = (count / total_samples) * 100
    print(f"{disease}: {count} samples ({percentage:.2f}%)")

# Data info
print("\nData Types:")
print(train_data.dtypes.unique())

# Top features (most common symptoms)
print("\n\nTop 10 Most Common Symptoms:")
feature_sums = train_data[feature_columns].sum().sort_values(ascending=False)
for i, (symptom, count) in enumerate(feature_sums.head(10).items(), 1):
    percentage = (count / total_samples) * 100
    print(f"{i}. {symptom}: appears in {count} patients ({percentage:.2f}%)")

# Least common symptoms
print("\n\nTop 10 Least Common Symptoms:")
for i, (symptom, count) in enumerate(feature_sums.tail(10).items(), 1):
    percentage = (count / total_samples) * 100
    print(f"{i}. {symptom}: appears in {count} patients ({percentage:.2f}%)")

print("\nEDA Complete!")
