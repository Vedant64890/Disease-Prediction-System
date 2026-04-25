
# Power BI Integration Guide

## Available Data Files for Power BI

### 1. powerbi_disease_symptom_mapping.csv
- **Purpose:** Disease-Symptom relationships
- **Columns:** Disease, Symptom, Prevalence_Score, Symptom_Count
- **Use Case:** Build heat maps, bubble charts showing disease-symptom relationships
- **Visualization:** Matrix visualization, Network diagram

### 2. powerbi_symptom_relationships.csv
- **Purpose:** Symptom co-occurrence patterns
- **Columns:** Symptom_1, Symptom_2, Correlation, Relationship_Strength
- **Use Case:** Show which symptoms appear together frequently
- **Visualization:** Network graph, Chord diagram

### 3. powerbi_feature_importance.csv
- **Purpose:** ML model feature importance
- **Columns:** Symptom, Importance_Score, Importance_Percentage, Rank
- **Use Case:** Identify most predictive symptoms
- **Visualization:** Bar chart, Waterfall chart, Gauge

### 4. powerbi_disease_statistics.csv
- **Purpose:** Disease-level statistics
- **Columns:** Disease, Disease_Code, Sample_Count, Model_Accuracy, Confidence_Score
- **Use Case:** Compare diseases, performance metrics
- **Visualization:** Table, Card, KPI visual

### 5. powerbi_symptom_statistics.csv
- **Purpose:** Symptom-level prevalence and occurrence
- **Columns:** Symptom, Symptom_Code, Prevalence_Percentage, Occurrence_Count
- **Use Case:** Show symptom distribution, frequency
- **Visualization:** Bar chart, Pie chart, Maps

### 6. powerbi_model_performance.csv
- **Purpose:** Overall model performance metrics
- **Columns:** Metric, Value
- **Use Case:** Dashboard KPIs, scorecards
- **Visualization:** Card, Gauge, KPI

### 7. powerbi_prediction_scenarios.csv
- **Purpose:** Sample predictions from test data
- **Columns:** Sample_ID, Symptom_Count, Predicted_Disease, Confidence_Score
- **Use Case:** Show real predictions, confidence trends
- **Visualization:** Scatter, Line, Table

## Recommended Dashboard Layout

### Page 1: Executive Summary
- Model Accuracy KPI
- Total Diseases & Symptoms
- Top 10 Most Predictive Symptoms (Feature Importance)

### Page 2: Disease Analytics
- Disease distribution table
- Disease to symptom heatmap
- Disease-specific statistics

### Page 3: Symptom Analysis
- Symptom prevalence ranking
- Symptom co-occurrence network
- Symptom statistics table

### Page 4: Model Performance
- Accuracy metrics
- Confidence score distribution
- Prediction success rate

### Page 5: Relationships
- Symptom relationship network
- Disease-symptom correlation matrix
- Interactive drill-down capabilities

## Integration Steps

1. Download all CSV files from the csv_files folder
2. Import each CSV as a data source in Power BI Desktop
3. Create relationships between tables using common identifiers
4. Build visualizations using recommended chart types
5. Create interactive filters for disease, symptom, and severity level
6. Set up automatic refresh if connecting to live data source

## Data Refresh Strategy

- Regenerate CSV files whenever new training data is available
- Keep model_metadata.pkl in sync with latest model
- Update feature importance after model retraining
- Refresh prediction scenarios quarterly for dashboard accuracy

---
Generated: 2026-04-08
System: Disease Prediction Dashboard
