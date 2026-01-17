"""
Training pipeline with cross-validation and learning curve hooks.
Added LightGBM option and better feature handling.
"""
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Try to import LightGBM (optional but recommended)
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM not available. Using Logistic Regression only.")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODEL_DIR = RESULTS_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
USE_LIGHTGBM = False  # Set to True to use LightGBM instead of LogReg


def load_training_set(path: Path = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load engineered training features and target; one-hot encode object columns."""
    path = path or (DATA_DIR / "application_train_features.csv")
    df = pd.read_csv(path)
    if "TARGET" not in df.columns:
        raise ValueError("TARGET column missing; ensure features include labels.")
    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])

    cat_cols = X.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

    # Replace infs and fill remaining NaNs
    X = X.replace([float("inf"), float("-inf")], np.nan).fillna(0)
    
    print(f"Loaded training data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def make_model(use_lgbm: bool = False):
    """Create model - either LightGBM or Logistic Regression."""
    
    if use_lgbm and LGBM_AVAILABLE:
        print("Using LightGBM model")
        return lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    else:
        print("Using Logistic Regression model")
        log_reg = LogisticRegression(
            max_iter=3000,
            n_jobs=-1,
            solver="lbfgs",
            class_weight="balanced",
            C=0.1,  # Regularization strength (lower = more regularization)
        )
        return make_pipeline(StandardScaler(with_mean=False), log_reg)


def cross_validate(model, X, y, n_splits: int = 5, random_state: int = 42):
    """Run stratified CV and return fold AUCs with fold-level progress output."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"[CV] Fold {fold_idx}/{n_splits} start...")
        model_fold = clone(model)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model_fold.fit(X_train, y_train)
        probas = model_fold.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, probas)
        scores.append(fold_auc)
        print(f"[CV] Fold {fold_idx}/{n_splits} AUC={fold_auc:.4f}")
    return scores


def plot_learning_curve(model, X, y, output_path: Path = None, max_samples: int = 120_000):
    """Generate a learning curve plot (train vs validation AUC) with controlled sample size to avoid OOM."""
    # Downsample for curve computation to limit memory usage
    if len(X) > max_samples:
        X, _, y, _ = train_test_split(X, y, train_size=max_samples, stratify=y, random_state=42)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=[0.1, 0.5, 1.0],
        cv=3,
        scoring="roc_auc",
        n_jobs=1,  # limit parallelism to reduce memory pressure
        shuffle=True,
        random_state=42,
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, label="Train AUC", marker="o")
    plt.plot(train_sizes, val_mean, label="Val AUC", marker="s")
    plt.xlabel("Training set size")
    plt.ylabel("AUC")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    output_path = output_path or (MODEL_DIR / "learning_curve.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def train_and_save():
    """End-to-end training: load data, CV, plot learning curve, persist model."""
    print("=" * 60)
    print("CREDIT SCORING MODEL TRAINING")
    print("=" * 60)
    
    X, y = load_training_set()
    model = make_model(use_lgbm=USE_LIGHTGBM)
    
    print(f"\nClass distribution: {y.value_counts().to_dict()}")
    print(f"Default rate: {y.mean():.2%}")
    
    aucs = cross_validate(model, X, y)
    mean_auc = sum(aucs) / len(aucs)
    std_auc = np.std(aucs)
    print(f"\nCV Results:")
    print(f"  AUCs: {[f'{a:.4f}' for a in aucs]}")
    print(f"  Mean: {mean_auc:.4f} (+/- {std_auc:.4f})")
    
    print("\nGenerating learning curve...")
    plot_path = plot_learning_curve(model, X, y)
    
    print("\nTraining final model on full data...")
    fitted = clone(model).fit(X, y) if not (USE_LIGHTGBM and LGBM_AVAILABLE) else model.fit(X, y)
    
    model_path = MODEL_DIR / "baseline_logreg.joblib"
    joblib.dump(fitted, model_path)
    
    # Save feature names for later use
    feature_names_path = MODEL_DIR / "feature_names.txt"
    with open(feature_names_path, 'w') as f:
        for col in X.columns:
            f.write(f"{col}\n")
    
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Model saved to: {model_path}")
    print(f"  Learning curve: {plot_path}")
    print(f"  CV AUC: {mean_auc:.4f}")
    
    # Update model report with actual CV scores
    report_path = MODEL_DIR / "model_report.txt"
    if report_path.exists():
        with open(report_path, 'r') as f:
            content = f.read()
        # Append actual scores
        if "ACTUAL TRAINING RESULTS" not in content:
            with open(report_path, 'a') as f:
                f.write(f"\n\nACTUAL TRAINING RESULTS\n")
                f.write(f"{'=' * 30}\n")
                f.write(f"Cross-Validation AUCs: {[f'{a:.4f}' for a in aucs]}\n")
                f.write(f"Mean CV AUC: {mean_auc:.4f} (+/- {std_auc:.4f})\n")
                f.write(f"Number of features: {X.shape[1]}\n")
                f.write(f"Training samples: {X.shape[0]}\n")


if __name__ == "__main__":
    train_and_save()
