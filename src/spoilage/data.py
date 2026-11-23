"""
Data cleaning and preprocessing module for Cold Storage Spoilage Prediction.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def compute_data_sha256(filepath: str = "data/raw/train.csv") -> str:
    """
    Compute the SHA-256 hash of the original raw dataset file using hashlib.
    """
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"Training data file not found at: {filepath}")
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def clean_crop(series: pd.Series) -> pd.Series:
    """
    Standardize crop categorical values by trimming whitespace,
    lowercasing, and resolving known spelling variations/abbreviations.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "app.": "apple",
            "pot.": "potato",
            "oni.": "onion",
        })
    )


def clean_data(
    df: pd.DataFrame,
    drop_leakage: bool = True,
    drop_identifiers: bool = True,
    is_train: bool = True,
    leakage_cols: List[str] = None,
    drop_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Clean the cold storage dataset according to data-quality findings:
    - Removes duplicate rows (for training data).
    - Standardizes crop spelling variants to true levels ('apple', 'potato', 'onion').
    - Removes data leakage column ('spoilage_found_at_unloading').
    - Removes non-predictive identifiers ('record_id', 'date').
    """
    df_clean = df.copy()

    # 1. Remove duplicate rows if in training mode
    if is_train:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # 2. Standardize crop spelling variants
    if "crop" in df_clean.columns:
        df_clean["crop"] = clean_crop(df_clean["crop"])

    # 3. Handle data leakage column
    leakage_cols = leakage_cols or ["spoilage_found_at_unloading"]
    if drop_leakage:
        for col in leakage_cols:
            if col in df_clean.columns:
                df_clean = df_clean.drop(columns=[col])

    # 4. Remove identifier columns
    drop_cols = drop_cols or ["record_id", "date"]
    if drop_identifiers:
        for col in drop_cols:
            if col in df_clean.columns:
                df_clean = df_clean.drop(columns=[col])

    # 5. Ensure no duplicate records remain in training data
    if is_train:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    return df_clean


def get_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer for numerical scaling and categorical encoding.
    """
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_cols,
            ),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )


def load_train_data(
    filepath: str,
    params: dict,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load raw training data, clean it, and return feature matrix X and target y.
    """
    raw_df = pd.read_csv(filepath)
    clean_df = clean_data(
        raw_df,
        drop_leakage=True,
        drop_identifiers=True,
        is_train=True,
        leakage_cols=params["data"].get("leakage_cols"),
        drop_cols=params["data"].get("drop_cols"),
    )
    target_col = params["data"]["target_col"]
    if target_col not in clean_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")

    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]
    return X, y
