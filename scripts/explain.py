"""
Explainability script using SHAP.
Generates visualizations for selected clients.
"""
import matplotlib.pyplot as plt
import pandas as pd
import shap
import joblib
import numpy as np
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "results" / "model"
OUTPUT_DIR = BASE_DIR / "results" / "clients_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load train and test features."""
    print("Loading data...")
    # Load train features
    train_df = pd.read_csv(DATA_DIR / "application_train_features.csv")
    if "TARGET" in train_df.columns:
        y_train = train_df["TARGET"]
        X_train = train_df.drop(columns=["TARGET"])
    else:
        X_train = train_df
        y_train = None
        
    # Load test features
    test_df = pd.read_csv(DATA_DIR / "application_test_features.csv")
    # Test data usually doesn't have TARGET in this context, or it's separate
    if "TARGET" in test_df.columns:
        X_test = test_df.drop(columns=["TARGET"])
    else:
        X_test = test_df
        
    # Handle categorical columns (One-Hot Encoding) as done in training
    # We need to align columns between train and test
    # For simplicity, we'll assume the features file is already preprocessed or we do minimal processing
    # In train.py: pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        X_train = pd.get_dummies(X_train, columns=cat_cols, dummy_na=True)
        X_test = pd.get_dummies(X_test, columns=cat_cols, dummy_na=True)
    
    # Align columns
    # We need to match the model's expected features exactly
    model_path = MODEL_DIR / "baseline_logreg.joblib"
    try:
        pipeline = joblib.load(model_path)
        # Try to get feature names from the first step (StandardScaler) or the pipeline itself if it exposes it
        if hasattr(pipeline, "feature_names_in_"):
            expected_features = pipeline.feature_names_in_
        elif hasattr(pipeline.steps[0][1], "feature_names_in_"):
            expected_features = pipeline.steps[0][1].feature_names_in_
        else:
            print("Warning: Could not find feature_names_in_ in model. Proceeding with current columns.")
            expected_features = None
            
        if expected_features is not None:
            print(f"Aligning to {len(expected_features)} expected features...")
            # Use reindex to align columns, filling missing with 0 and dropping extras
            X_train = X_train.reindex(columns=expected_features, fill_value=0)
            X_test = X_test.reindex(columns=expected_features, fill_value=0)
            
    except Exception as e:
        print(f"Error aligning features: {e}")
        import traceback
        traceback.print_exc()
    
    # Fill NaNs as in train.py
    X_train = X_train.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    X_test = X_test.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    
    return X_train, y_train, X_test

def load_model():
    """Load the trained pipeline."""
    model_path = MODEL_DIR / "baseline_logreg.joblib"
    print(f"Loading model from {model_path}...")
    return joblib.load(model_path)

def generate_shap_plots(model, X_train, X_test):
    """Generate SHAP plots for specific clients."""
    
    # The model is a Pipeline (StandardScaler -> LogisticRegression)
    # We need to explain the LogisticRegression part, but pass scaled data
    
    pipeline = model
    scaler = pipeline.named_steps['standardscaler']
    classifier = pipeline.named_steps['logisticregression']
    
    # Prepare background data for SHAP (using a summary of train data to save time)
    # For linear models, we can use the coefficients directly or LinearExplainer
    # LinearExplainer is faster and exact for linear models
    
    # We need to transform X_train using the scaler first
    print("Scaling data for SHAP...")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for better feature names in SHAP
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print("Initializing SHAP Explainer...")
    explainer = shap.LinearExplainer(classifier, X_train_scaled_df)
    
    # Select clients
    # 2 from train: one correct, one wrong (we need predictions for this)
    # 1 from test
    
    print("Selecting clients...")
    # Predict on a subset of train to find correct/wrong examples
    # (Doing full predict might be slow, let's take first 1000)
    subset_size = 1000
    train_subset = X_train_scaled_df.iloc[:subset_size]
    train_subset_orig = X_train.iloc[:subset_size]
    
    # Get predictions
    probs = classifier.predict_proba(train_subset)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    # We need y_train for this. 
    # Note: In load_data we returned y_train. We need to make sure we use the aligned one if we did alignment?
    # Actually y_train corresponds to X_train rows.
    
    # Let's re-fetch y_train corresponding to the subset
    # We need to pass y_train to this function or reload it. 
    # Let's assume y_train is available or we pass it.
    # For now, I'll just pick random indices and assume their state for demonstration if y is not passed.
    # Wait, I can't know if it's correct/wrong without y.
    
    # Let's just pick:
    # Client 1: High probability (likely default)
    # Client 2: Low probability (likely repay)
    # Client 3: From test set
    
    # If we want strictly "correct" and "wrong", we need labels.
    # I will modify the function signature to accept y_train.
    pass

def run_explanation():
    X_train, y_train, X_test = load_data()
    model = load_model()
    
    pipeline = model
    scaler = pipeline.named_steps['standardscaler']
    classifier = pipeline.named_steps['logisticregression']
    
    # Scale data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Predictions on train subset to find examples
    subset_idx = 0
    subset_size = 2000
    
    train_subset_X = X_train_scaled_df.iloc[:subset_size]
    train_subset_y = y_train.iloc[:subset_size].values
    
    probs = classifier.predict_proba(train_subset_X)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    # Find one correct and one wrong
    correct_indices = np.where(preds == train_subset_y)[0]
    wrong_indices = np.where(preds != train_subset_y)[0]
    
    if len(correct_indices) > 0:
        client_correct_idx = correct_indices[0]
    else:
        client_correct_idx = 0 # Fallback
        
    if len(wrong_indices) > 0:
        client_wrong_idx = wrong_indices[0]
    else:
        client_wrong_idx = 1 # Fallback
        
    client_test_idx = 0 # First client in test set
    
    print(f"Selected Client (Correct): Index {client_correct_idx}, True: {train_subset_y[client_correct_idx]}, Pred: {preds[client_correct_idx]}")
    print(f"Selected Client (Wrong): Index {client_wrong_idx}, True: {train_subset_y[client_wrong_idx]}, Pred: {preds[client_wrong_idx]}")
    print(f"Selected Client (Test): Index {client_test_idx}")
    
    explainer = shap.LinearExplainer(classifier, X_train_scaled_df)
    
    # Generate plots
    clients = [
        ("client_correct_train", X_train_scaled_df.iloc[[client_correct_idx]]),
        ("client_wrong_train", X_train_scaled_df.iloc[[client_wrong_idx]]),
        ("client_test", X_test_scaled_df.iloc[[client_test_idx]])
    ]
    
    for name, client_data in clients:
        print(f"Generating SHAP for {name}...")
        shap_values = explainer.shap_values(client_data)
        
        # Force plot
        # shap.force_plot returns HTML, we can save it
        p = shap.force_plot(
            explainer.expected_value, 
            shap_values, 
            client_data, 
            matplotlib=True, 
            show=False
        )
        plt.savefig(OUTPUT_DIR / f"{name}_force_plot.png", bbox_inches='tight', dpi=150)
        plt.close()
        
        # Waterfall plot (better for single prediction)
        plt.figure()
        # shap.plots.waterfall needs an Explanation object, LinearExplainer returns arrays for shap_values usually
        # Let's construct an Explanation object or use summary_plot for single instance (bar)
        
        # For older shap versions or simple array output:
        # We can use summary_plot with plot_type="bar" for a single instance
        shap.summary_plot(shap_values, client_data, plot_type="bar", show=False)
        plt.savefig(OUTPUT_DIR / f"{name}_summary_bar.png", bbox_inches='tight')
        plt.close()

    print("Done! Check results/clients_outputs/")

if __name__ == "__main__":
    run_explanation()
