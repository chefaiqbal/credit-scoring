"""
Generate Kaggle Submission File
===============================
Creates a properly formatted submission file for the Home Credit Default Risk competition.
"""
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "results" / "model"
OUTPUT_DIR = BASE_DIR / "results"

def load_model():
    """Load the trained pipeline."""
    model_path = MODEL_DIR / "baseline_logreg.joblib"
    print(f"Loading model from {model_path}...")
    return joblib.load(model_path)

def load_test_features():
    """Load test features."""
    test_path = DATA_DIR / "application_test_features.csv"
    print(f"Loading test features from {test_path}...")
    df = pd.read_csv(test_path)
    
    # Store SK_ID_CURR for submission
    sk_ids = df["SK_ID_CURR"].values
    
    return df, sk_ids

def prepare_features(df, pipeline):
    """Prepare features to match model expectations."""
    X = df.copy()
    
    # Remove SK_ID_CURR for prediction
    if "SK_ID_CURR" in X.columns:
        X = X.drop(columns=["SK_ID_CURR"])
    
    # Remove TARGET if present
    if "TARGET" in X.columns:
        X = X.drop(columns=["TARGET"])
    
    # One-hot encode categoricals
    cat_cols = X.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    
    # Align with model features
    scaler = pipeline.named_steps['standardscaler']
    if hasattr(pipeline, "feature_names_in_"):
        expected_features = pipeline.feature_names_in_
    elif hasattr(scaler, "feature_names_in_"):
        expected_features = scaler.feature_names_in_
    else:
        expected_features = X.columns
    
    print(f"Aligning to {len(expected_features)} model features...")
    X = X.reindex(columns=expected_features, fill_value=0)
    
    # Clean data
    X = X.replace([float("inf"), float("-inf")], np.nan).fillna(0)
    
    return X

def generate_submission():
    """Generate Kaggle submission file."""
    print("=" * 60)
    print("KAGGLE SUBMISSION GENERATOR")
    print("=" * 60)
    
    # Load model and data
    pipeline = load_model()
    test_df, sk_ids = load_test_features()
    
    # Prepare features
    X_test = prepare_features(test_df, pipeline)
    print(f"Test data shape: {X_test.shape}")
    
    # Predict
    print("Generating predictions...")
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        "SK_ID_CURR": sk_ids,
        "TARGET": probabilities
    })
    
    # Save submission
    submission_path = OUTPUT_DIR / "kaggle_submission.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"\n{'=' * 60}")
    print("SUBMISSION GENERATED")
    print(f"{'=' * 60}")
    print(f"File saved to: {submission_path}")
    print(f"Samples: {len(submission)}")
    print(f"Prediction range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
    print(f"Mean prediction: {probabilities.mean():.4f}")
    
    # Show sample
    print(f"\nSample predictions:")
    print(submission.head(10).to_string(index=False))
    
    return submission_path

if __name__ == "__main__":
    generate_submission()
