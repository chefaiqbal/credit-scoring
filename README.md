# 🏦 Credit Scoring Project

## 📋 Overview
This project implements a **credit scoring model** to predict the probability of default for clients. It uses a **Logistic Regression** model trained on the Home Credit Default Risk dataset. The project includes exploratory data analysis, a trained model, a reporting dashboard, and interpretability analysis using SHAP.

## 🚀 How to Run the Code

### 📦 Prerequisites
Install the required libraries:
```bash
pip install -r requirements.txt
```

### 1️⃣ Feature Engineering & Training
To preprocess data and train the model:
```bash
python3 scripts/preprocess.py
python3 scripts/train.py
```

### 2️⃣ Prediction
To run predictions on the test set and see the AUC score:
```bash
python3 scripts/predict.py
```

### 3️⃣ Dashboard
To launch the interactive dashboard:
```bash
streamlit run results/dashboard/dashboard.py
```

### 4️⃣ Explainability
To generate SHAP plots and client reports:
```bash
python3 scripts/explain.py
```

## 📂 Project Structure
- 🗂️ `data/`: Contains the dataset files.
- 📊 `results/`: Contains model artifacts, EDA notebooks, client outputs, and the dashboard.
- 📜 `scripts/`: Contains Python scripts for preprocessing, training, prediction, and explanation.

