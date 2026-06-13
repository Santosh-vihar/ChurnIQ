"""
Preprocessing utilities for ChurnIQ.
Shared between training and inference so the pipeline stays consistent.
"""

import pandas as pd
import numpy as np


# Columns to drop before training (identifiers, not features)
DROP_COLS = ["customerID"]

# Target column
TARGET_COL = "Churn"

# Categorical columns that need encoding
CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Numeric columns
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning steps:
    - Strip whitespace from column names and string values
    - Fix TotalCharges (sometimes stored as string with spaces)
    - Drop rows with missing TotalCharges after coercion
    """
    df = df.copy()

    # Trim column names
    df.columns = df.columns.str.strip()

    # TotalCharges can have ' ' (space) for new customers — coerce to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows where TotalCharges is NaN (11 records in the standard dataset)
    df.dropna(subset=["TotalCharges"], inplace=True)

    # Reset index after dropping rows
    df.reset_index(drop=True, inplace=True)

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Churn Yes/No → 1/0."""
    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of feature columns in the order used for training.
    Excludes the target and identifier columns.
    """
    exclude = DROP_COLS + [TARGET_COL]
    return [c for c in df.columns if c not in exclude]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw DataFrame for the sklearn pipeline:
    - Drop identifier columns
    - Return feature-only DataFrame (no target)
    """
    df = df.copy()
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if TARGET_COL in df.columns:
        cols_to_drop.append(TARGET_COL)
    df.drop(columns=cols_to_drop, inplace=True)
    return df
