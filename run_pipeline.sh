#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting Credit Scoring Pipeline..."

echo "--------------------------------------------------"
echo "1️⃣  Running Feature Engineering (Preprocess)..."
python3 scripts/preprocess.py

echo "--------------------------------------------------"
echo "2️⃣  Training Model..."
python3 scripts/train.py

echo "--------------------------------------------------"
echo "3️⃣  Running Predictions..."
python3 scripts/predict.py

echo "--------------------------------------------------"
echo "4️⃣  Generating Explanations (SHAP)..."
python3 scripts/explain.py

echo "--------------------------------------------------"
echo "✅ Pipeline Completed Successfully!"
echo "   - Model artifacts in results/model/"
echo "   - Client reports in results/clients_outputs/"
echo "   - To view the dashboard, run: streamlit run results/dashboard/dashboard.py"
