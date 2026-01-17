"""
Training pipeline placeholder with cross-validation and learning curve hooks.
"""
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODEL_DIR = RESULTS_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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
    X = X.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    return X, y


def make_model():
    """Create a baseline model with scaling and higher max_iter."""
    log_reg = LogisticRegression(
        max_iter=3000,
        n_jobs=-1,
        solver="lbfgs",
        class_weight="balanced",
    )
    # StandardScaler with mean disabled to support sparse-ish inputs safely
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
    """End-to-end training stub: load data, CV, plot learning curve, persist model."""
    X, y = load_training_set()
    model = make_model()
    aucs = cross_validate(model, X, y)
    print(f"CV AUCs: {aucs}; mean={sum(aucs)/len(aucs):.4f}")
    plot_path = plot_learning_curve(model, X, y)
    fitted = model.fit(X, y)
    model_path = MODEL_DIR / "baseline_logreg.joblib"
    joblib.dump(fitted, model_path)
    print(f"Saved model to {model_path}")
    print(f"Saved learning curve to {plot_path}")


if __name__ == "__main__":
    train_and_save()
