"""
train_model.py — ChurnIQ model training script

Run this once to train the XGBoost classifier and save all artifacts.
Usage:
    python train_model.py

Outputs (saved to artifacts/):
    - pipeline.joblib     : full sklearn Pipeline (preprocessor + XGBoost)
    - metadata.joblib     : feature names, class labels, model eval metrics
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from xgboost import XGBClassifier

from utils.preprocessing import (
    load_raw_data,
    clean_data,
    encode_target,
    get_feature_columns,
    prepare_features,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
)

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join("data", "telco_customer_churn.csv")
ARTIFACTS_DIR = "artifacts"


def build_preprocessor():
    """
    Build a ColumnTransformer that:
    - Ordinal-encodes all categorical columns
    - Scales numeric columns with StandardScaler
    """
    categorical_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_COLS),
            ("num", numeric_transformer, NUMERIC_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def build_pipeline(preprocessor):
    """Combine preprocessor with XGBoost in a single sklearn Pipeline."""
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.7,   # roughly ~73% No / ~27% Yes
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", xgb)])
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """Return a dict with all key classification metrics."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }
    return metrics


def get_feature_importance(pipeline, feature_names):
    """Extract feature importances from the XGBoost step."""
    xgb_model = pipeline.named_steps["classifier"]
    importances = xgb_model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    return importance_df.to_dict(orient="list")


def save_artifacts(pipeline, metadata):
    """Save the pipeline and metadata dict to the artifacts/ directory."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    pipeline_path = os.path.join(ARTIFACTS_DIR, "pipeline.joblib")
    metadata_path = os.path.join(ARTIFACTS_DIR, "metadata.joblib")

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(metadata, metadata_path)

    print(f"  Saved pipeline  → {pipeline_path}")
    print(f"  Saved metadata  → {metadata_path}")


def main():
    print("=" * 55)
    print("  ChurnIQ — Model Training")
    print("=" * 55)

    # --- Load & clean ---
    print("\n[1/5] Loading data ...")
    df = load_raw_data(DATA_PATH)
    df = clean_data(df)
    df = encode_target(df)
    print(f"      Dataset shape: {df.shape}  |  Churn rate: {df[TARGET_COL].mean():.2%}")

    # --- Feature / target split ---
    print("\n[2/5] Preparing features ...")
    feature_cols = get_feature_columns(df)   # ordered list, no ID / target
    X = prepare_features(df)
    y = df[TARGET_COL]

    # Reorder columns to match CATEGORICAL_COLS + NUMERIC_COLS expected by ColumnTransformer
    ordered_cols = [c for c in CATEGORICAL_COLS if c in X.columns] + \
                   [c for c in NUMERIC_COLS if c in X.columns]
    X = X[ordered_cols]
    feature_names = ordered_cols  # names after reorder

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # --- Build & train ---
    print("\n[3/5] Training XGBoost pipeline ...")
    preprocessor = build_preprocessor()
    pipeline = build_pipeline(preprocessor)
    pipeline.fit(X_train, y_train)
    print("      Training complete.")

    # --- Evaluate ---
    print("\n[4/5] Evaluating model ...")
    metrics = evaluate_model(pipeline, X_test, y_test)
    print(f"      Accuracy  : {metrics['accuracy']:.4f}")
    print(f"      Precision : {metrics['precision']:.4f}")
    print(f"      Recall    : {metrics['recall']:.4f}")
    print(f"      F1 Score  : {metrics['f1_score']:.4f}")
    print(f"      ROC-AUC   : {metrics['roc_auc']:.4f}")

    # --- Feature importance ---
    importance = get_feature_importance(pipeline, feature_names)

    # --- Build metadata dict ---
    metadata = {
        "feature_names": feature_names,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "target_col": TARGET_COL,
        "metrics": metrics,
        "feature_importance": importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "churn_rate": float(df[TARGET_COL].mean()),
    }

    # --- Save ---
    print("\n[5/5] Saving artifacts ...")
    save_artifacts(pipeline, metadata)

    print("\n  Done! Run the app with:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
