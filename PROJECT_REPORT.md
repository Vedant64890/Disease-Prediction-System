# Disease Prediction System — Project Report

**Date:** 2026-04-10  
**Project Type:** End-to-end ML pipeline + Interactive Prediction App + (Planned) Power BI Integration

---

## 1) Abstract
This project implements a symptom-to-disease prediction workflow using classical machine learning on a structured symptom dataset. It includes a reproducible preprocessing stage, an exploratory data analysis (EDA) script, model training with validation-based model selection, test-set evaluation, and two user-facing interfaces for inference:
1) a CLI-driven prediction program, and 2) a Streamlit dashboard application with user authentication, interactive symptom selection, prediction confidence breakdowns, and symptom/remedy guidance.  
A separate guide outlines how the system’s analytics can be presented in Power BI via CSV datasets.

---

## 2) Description
### 2.1 Problem Statement
Given a set of symptoms (represented as binary indicators), predict the most likely disease class and provide supporting analytics (confidence distribution, ranked alternatives, and simple interpretability cues).

### 2.2 Data Sources
- **Training dataset:** `csv_files/Training.csv`
- **Testing dataset:** `csv_files/Testing.csv`
- **Symptom → home remedy lookup:** `csv_files/home_remedies_training.csv`

**Data shape conventions (based on code):**
- Each row represents a patient/sample.
- Feature columns represent symptoms (typically `0/1`).
- Target column is `prognosis` (disease label).

### 2.3 System Workflow (High-Level)
1. **Preprocessing**: Load → clean → align train/test columns → encode target labels → stratified train/validation split → save artifacts.
2. **EDA**: Print dataset characteristics, disease distribution, feature occurrence statistics.
3. **Training**: Train multiple models, compare validation metrics, save best model + metadata.
4. **Evaluation**: Evaluate the chosen model on the test set; produce detailed diagnostics.
5. **Prediction Interfaces**:
   - CLI program for demo/interactive/quick prediction.
   - Streamlit dashboard with login and interactive charts + remedy info.
6. **Power BI** (documented): A guide describes intended CSV outputs and recommended dashboard layout.

---

## 3) Key Features
### 3.1 Data Preparation & Feature Engineering
- Drops duplicate rows.
- Removes unnamed columns.
- Trims/standardizes column names.
- Fills missing values with `0`.
- Aligns test-set features to the training feature set.
- Encodes disease labels using `LabelEncoder`.
- Stratified train/validation split to preserve class distribution.

### 3.2 Exploratory Data Analysis (EDA)
- Prints dataset shapes and column lists.
- Disease class distribution and class balance percentages.
- Summary statistics for symptom features.
- Top 10 most common symptoms and top 10 least common symptoms.

### 3.3 Model Training & Selection
- Trains multiple candidate models:
  - **K-Nearest Neighbors** classifier.
  - **Linear Regression** (converted to a pseudo-classifier by rounding/clipping predictions).
- Compares models using:
  - Training accuracy
  - Validation accuracy
  - Weighted precision, recall, F1
- Saves the best model to `best_model.pkl` with supporting metadata (feature names, encoder, metrics).

### 3.4 Test Evaluation & Diagnostics
- Evaluates on the test set and prints:
  - Accuracy, precision, recall, F1
  - Full classification report
  - Confusion matrix analysis and top misclassifications
  - Per-class performance table (best/worst classes)
  - A “Validation vs Test” summary

### 3.5 CLI Prediction Application
- Offers three modes:
  - **Demo mode** (pre-filled symptom patterns)
  - **Interactive mode** (choose symptoms by number)
  - **Quick mode** (type symptom names)
- Produces:
  - Primary predicted disease
  - Confidence score
  - Top-5 predictions with a bar visualization in the console

### 3.6 Streamlit Dashboard Application
- Multi-page navigation with user authentication:
  - Home, About, Register, Login, Contact Us
  - Dashboard page is accessible only after login
- Prediction UI:
  - Symptom multi-select (with selection cap)
  - One-click prediction
  - Result cards: disease, confidence, symptom count
  - “Find Hospital” external link based on predicted disease
- Visual analytics:
  - Bar chart (selected symptoms or influence scores when supported)
  - Pie chart showing predicted probability distribution
  - Ranked prediction table (top-10)
- Guidance:
  - Symptom cue table (basic indication text)
  - Symptom-wise home remedies (from dataset)
  - Disease “cure info” panel (home remedies/medicines/doctor required)

### 3.7 Power BI Integration (Planned / Documented)
The project includes a guide describing intended Power BI datasets such as:
- disease–symptom mapping
- symptom relationships
- feature importance
- disease/symptom statistics
- model performance and prediction scenarios

**Note:** In the current workspace, these Power BI CSVs are not present and there is no script yet that generates them.

---

## 4) Concepts Used
### 4.1 Data Engineering
- Data loading and cleaning with **pandas**.
- Feature alignment between training and testing datasets.
- Handling missing values and duplicates.

### 4.2 Machine Learning
- Supervised multi-class classification framing.
- Train/validation split with **stratification**.
- **Label Encoding** for categorical targets.
- Baseline models:
  - **KNN** for classification on binary symptom vectors.
  - **Linear Regression** adapted to discrete classes (round + clip).
- Model selection via validation metrics.

### 4.3 Model Evaluation
- Classification metrics:
  - Accuracy
  - Weighted Precision / Recall / F1
- Confusion matrix-based error analysis.
- Per-class performance ranking.

### 4.4 Probability Handling
- Uses `predict_proba` when available.
- Implements fallbacks for models without probabilities (one-hot probability for predicted class).

### 4.5 Application & UI Engineering (Streamlit)
- Stateful navigation via `st.session_state`.
- Caching:
  - `st.cache_resource` for model/data
  - `st.cache_data` for symptom remedy CSV
- Interactive input widgets:
  - Radio navigation, forms, multi-select, buttons
- Visualization with **Plotly** (bar and pie charts).

### 4.6 Authentication & Security (Basic)
- User storage in a JSON file.
- Password hashing using **PBKDF2-HMAC-SHA256** with per-user salts.
- Constant-time comparisons (`hmac.compare_digest`).

---

## 5) UI (User Interface) Details

### 5.1 CLI UI (Terminal)
**Entry point:** prediction script provides a menu:
- Demo Mode
- Interactive Mode (symptom numbers)
- Quick Mode (symptom names)
- View All Symptoms
- Exit

**Output style:** formatted headers, ranked predictions, confidence bars.

### 5.2 Streamlit Web UI
**App title:** “Disease Prediction System”

**Navigation behavior:**
- Not logged-in: Home / About / Register / Login / Contact Us
- Logged-in: Home / Dashboard / About / Contact Us (plus Logout)

**Dashboard page sections:**
1. Prediction section (symptom selection → predict)
2. Charts (bar and pie)
3. Ranked predictions table
4. Symptom cues and remedy tables
5. Cure guidance panel (home remedies + medicines + doctor requirement)

**Important UX note:** Dashboard access is gated by login.

---

## 6) Outputs

### 6.1 Generated Files (Artifacts)
Running the pipeline generates these primary artifacts:
- `preprocessing_data.pkl` — encoded labels, splits, test features, feature names
- `best_model.pkl` — best model + metadata for inference

### 6.2 Console Outputs (Examples)
Scripts print:
- Dataset shapes, column names, disease classes
- Training/validation metrics and comparison table
- Test-set metrics, confusion matrix findings
- Prediction results and ranked confidence outputs

### 6.3 Dashboard Outputs
- On-screen metrics (total diseases/symptoms, accuracy, samples)
- Interactive charts and tables
- Hospital search link based on predicted disease

### 6.4 Power BI Outputs (As Described in Guide)
The guide lists intended Power BI CSV datasets such as:
- `powerbi_model_performance.csv`
- `powerbi_feature_importance.csv`
- `powerbi_disease_statistics.csv`
- `powerbi_symptom_statistics.csv`
- `powerbi_prediction_scenarios.csv`

**Current status:** those CSV files are not present in `csv_files/` yet.

---

## 7) How to Run (Typical Sequence)
1. Preprocess: `python s01_prep.py`
2. EDA: `python s02_eda.py`
3. Train: `python s03_train.py`
4. Evaluate: `python s04_eval.py`
5. Predict (CLI): `python s05_predict.py`
6. Dashboard (Streamlit): `streamlit run s06_dash.py`

---

## 8) Assumptions, Limitations, and Notes
- Predictions are **not** medical diagnoses; output is for educational/screening support.
- Linear Regression is not a natural multi-class classifier; it’s included as a baseline and adapted via rounding/clipping.
- “Power BI integration” is currently documented, but the pipeline does not yet generate the described Power BI CSVs.
- The UI includes basic authentication via a JSON file; it is suitable for demos but not a production-grade user store.

---

## 9) Conclusion
This project provides a complete, runnable disease prediction pipeline with both CLI and web UI options, supported by standard ML evaluation practices and an analytics-oriented dashboard experience. With a future step to generate the Power BI CSV datasets described in the integration guide, it can also support external BI reporting workflows.
