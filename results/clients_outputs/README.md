# Client Explainability Analysis

This folder contains model interpretability outputs for three selected clients as required by the project specification.

## Generated Files

### For Each Client:
- `*_force_plot.png` - SHAP force plot showing feature contributions to prediction
- `*_shap_importance.html` - Interactive Plotly bar chart of top SHAP values
- `*_comparison.html` - Interactive Plotly visualization comparing client vs population

### Analysis Reports:
- `client_analysis_report.txt` - Master report with all client details
- `wrong_prediction_analysis.txt` - Detailed analysis of why the model failed

## Client Selection Criteria

### 1. `client_correct_train` (Train Set - Correct Prediction)
A client where the model correctly predicted the outcome. Prioritizes correctly identified defaulters with high confidence.

### 2. `client_wrong_train` (Train Set - Wrong Prediction)  
A client where the model's prediction was incorrect. Prioritizes **False Negatives** (predicted safe but actually defaulted) as these are the most dangerous errors in credit scoring.

### 3. `client_test` (Test Set)
A client from the test set to demonstrate prediction on unseen data.

## Understanding the Visualizations

### SHAP Force Plots
- **Red features**: Push prediction toward higher risk (default)
- **Blue features**: Push prediction toward lower risk (repay)
- The width of each bar indicates the magnitude of impact

### SHAP Importance Bar Charts
- Shows the top 15 most influential features for this specific prediction
- Positive values (red) increase default probability
- Negative values (green) decrease default probability

### Population Comparison Plots
- Histograms show the distribution of features across all clients
- Red dashed line indicates where this specific client falls
- Helps identify if client has unusual values

## Running the Analysis

```bash
python scripts/explain.py
```
