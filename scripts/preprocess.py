"""
Home Credit preprocessing pipeline.
Handles application data plus auxiliary tables (bureau, previous applications, installments, credit card, POS cash, etc.).
"""
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_CAT_CARDINALITY = 24  # avoid exploding one-hot width on high-cardinality columns
CHUNKSIZE = 100_000
INSTALLMENT_NUM_COLS = [
    "AMT_INSTALMENT",
    "AMT_PAYMENT",
    "DAYS_ENTRY_PAYMENT",
    "DAYS_INSTALMENT",
    "NUM_INSTALMENT_VERSION",
    "NUM_INSTALMENT_NUMBER",
]
CC_NUM_COLS = [
    "AMT_PAYMENT_TOTAL_CURRENT",
    "AMT_BALANCE",
    "AMT_CREDIT_LIMIT_ACTUAL",
    "SK_DPD",
    "SK_DPD_DEF",
    "MONTHS_BALANCE",  # recency proxy (closer to 0 = recent)
]
POS_NUM_COLS = [
    "AMT_INSTALMENT",
    "SK_DPD",
    "SK_DPD_DEF",
    "MONTHS_BALANCE",  # recency proxy
]


def load_application_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load application train and test data, adding a dataset flag for recombination."""
    train = pd.read_csv(DATA_DIR / "application_train.csv")
    test = pd.read_csv(DATA_DIR / "application_test.csv")
    train["dataset"] = "train"
    test["dataset"] = "test"
    return train, test


def auxiliary_table_paths() -> Dict[str, Path]:
    """Return paths to auxiliary raw tables."""
    return {
        "bureau": DATA_DIR / "bureau.csv",
        "previous_application": DATA_DIR / "previous_application.csv",
        "installments_payments": DATA_DIR / "installments_payments.csv",
        "credit_card_balance": DATA_DIR / "credit_card_balance.csv",
        "pos_cash_balance": DATA_DIR / "POS_CASH_balance.csv",
    }


def load_auxiliary_tables() -> Dict[str, pd.DataFrame]:
    """Load auxiliary tables into memory."""
    return {name: pd.read_csv(path) for name, path in auxiliary_table_paths().items() if path.exists()}


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _aggregate_numeric(df: pd.DataFrame, group_key: str, prefix: str) -> pd.DataFrame:
    """Aggregate numeric columns with lightweight statistics and prefix names."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if group_key not in numeric_cols:
        numeric_cols.append(group_key)
    agg_funcs = ["mean"]  # kept minimal to stay memory-efficient
    grouped = df[numeric_cols].groupby(group_key).agg(agg_funcs)
    grouped.columns = [f"{prefix}_{col}_{stat}" for col, stat in grouped.columns]
    grouped.reset_index(inplace=True)
    return grouped


def _count_rows_file(path: Path, group_key: str, prefix: str, chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    """Chunked row counts per group_key to produce lightweight features."""
    counts = None
    for chunk in pd.read_csv(path, usecols=[group_key], chunksize=chunksize):
        grp = chunk.groupby(group_key).size()
        counts = grp if counts is None else counts.add(grp, fill_value=0)
    if counts is None:
        return pd.DataFrame(columns=[group_key, f"{prefix}_row_count"])
    counts = counts.astype(int)
    return counts.rename(f"{prefix}_row_count").reset_index()


def _aggregate_categoricals(df: pd.DataFrame, group_key: str, prefix: str) -> pd.DataFrame:
    """Skip high-cardinality categoricals to keep the feature matrix compact."""
    return pd.DataFrame({group_key: df[group_key].unique()})


def _aggregate_means_file(path: Path, group_key: str, numeric_cols: list, prefix: str, chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    """Chunked mean aggregation for selected numeric columns."""
    # Detect available columns cheaply
    header = pd.read_csv(path, nrows=0)
    cols = [c for c in numeric_cols if c in header.columns]
    if group_key not in header.columns:
        return pd.DataFrame(columns=[group_key])
    if not cols:
        return pd.DataFrame(columns=[group_key])

    usecols = [group_key] + cols
    sum_df = None
    count_df = None

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = _downcast_numeric(chunk)
        grouped = chunk.groupby(group_key)[cols].agg(["sum", "count"])
        sum_part = grouped.xs("sum", level=1, axis=1)
        count_part = grouped.xs("count", level=1, axis=1)

        if sum_df is None:
            sum_df = sum_part
            count_df = count_part
        else:
            sum_df = sum_df.add(sum_part, fill_value=0)
            count_df = count_df.add(count_part, fill_value=0)

    if sum_df is None or count_df is None:
        return pd.DataFrame(columns=[group_key])

    mean_df = sum_df / count_df.replace(0, pd.NA)
    mean_df.reset_index(inplace=True)
    mean_df.columns = [f"{prefix}_{c}_mean" if c != group_key else group_key for c in mean_df.columns]
    return mean_df


def _merge(left: pd.DataFrame, right: pd.DataFrame, key: str) -> pd.DataFrame:
    """Helper to merge feature blocks safely."""
    if right is None or right.empty:
        return left
    return left.merge(right, on=key, how="left")


def build_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load application data and aggregate auxiliary tables into SK_ID_CURR level features.
    Returns train and test feature DataFrames.
    """
    app_train, app_test = load_application_data()
    aux_tables = load_auxiliary_tables()

    full = pd.concat([app_train, app_test], ignore_index=True, sort=False)

    bureau_path = auxiliary_table_paths().get("bureau")
    if bureau_path and bureau_path.exists():
        bureau_counts = _count_rows_file(bureau_path, "SK_ID_CURR", "bureau")
        full = _merge(full, bureau_counts, "SK_ID_CURR")

    # Previous applications
    prev_path = auxiliary_table_paths().get("previous_application")
    if prev_path and prev_path.exists():
        prev_counts = _count_rows_file(prev_path, "SK_ID_CURR", "prev")
        full = _merge(full, prev_counts, "SK_ID_CURR")

    # Installments payments
    inst_path = auxiliary_table_paths().get("installments_payments")
    if inst_path and inst_path.exists():
        inst_means = _aggregate_means_file(inst_path, "SK_ID_CURR", INSTALLMENT_NUM_COLS, "inst")
        full = _merge(full, inst_means, "SK_ID_CURR")

    # Credit card balance
    ccb_path = auxiliary_table_paths().get("credit_card_balance")
    if ccb_path and ccb_path.exists():
        ccb_means = _aggregate_means_file(ccb_path, "SK_ID_CURR", CC_NUM_COLS, "cc")
        full = _merge(full, ccb_means, "SK_ID_CURR")

    # POS cash balance
    pos_path = auxiliary_table_paths().get("pos_cash_balance")
    if pos_path and pos_path.exists():
        pos_means = _aggregate_means_file(pos_path, "SK_ID_CURR", POS_NUM_COLS, "pos")
        full = _merge(full, pos_means, "SK_ID_CURR")

    # Split back to train/test and drop helper flag
    train_features = full[full["dataset"] == "train"].drop(columns=["dataset"])
    test_features = full[full["dataset"] == "test"].drop(columns=["dataset", "TARGET"], errors="ignore")
    return train_features, test_features


def save_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Path, Path]:
    """Persist engineered features next to raw data."""
    train_path = DATA_DIR / "application_train_features.csv"
    test_path = DATA_DIR / "application_test_features.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def run_preprocessing() -> Tuple[Path, Path]:
    """Orchestrate preprocessing end-to-end for both train and test."""
    train_df, test_df = build_features()
    return save_features(train_df, test_df)


if __name__ == "__main__":
    train_out, test_out = run_preprocessing()
    print(f"Saved train features to {train_out}")
    print(f"Saved test features to {test_out}")
