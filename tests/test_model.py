"""
Tests for Model Training, Preprocessing, MLflow Tracking, and Registry.
These tests use in-memory synthetic test data so they do not depend on raw CSVs in CI.
"""

import hashlib
from pathlib import Path
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest

from src.spoilage.data import clean_crop, clean_data, get_preprocessor
from src.spoilage.train import build_pipeline, train_and_evaluate_model


@pytest.fixture
def sample_raw_dataframe():
    """
    Creates a small synthetic dataset mirroring train.csv data-quality issues:
    - duplicate rows
    - spelling variations in crop (e.g. 'app.', 'pot.', 'oni.', 'APPLE ')
    - data leakage column 'spoilage_found_at_unloading'
    - identifier columns 'record_id', 'date'
    """
    data = {
        "record_id": [f"REC{i:04d}" for i in range(1, 11)],
        "date": ["2026-01-15"] * 10,
        "storage_days": [10, 20, 30, 40, 50, 60, 70, 80, 20, 30],
        "temp_set_c": [2.0, 3.0, 4.0, 2.5, 3.5, 4.5, 2.0, 3.0, 3.0, 4.0],
        "humidity_pct": [85.0, 90.0, 80.0, 88.0, 92.0, 84.0, 86.0, 89.0, 90.0, 80.0],
        "power_cuts_hours": [5.0, 10.0, 2.0, 8.0, 15.0, 3.0, 7.0, 12.0, 10.0, 2.0],
        "bag_weight_kg": [50.0, 45.0, 55.0, 50.0, 48.0, 52.0, 49.0, 47.0, 45.0, 55.0],
        "door_openings_per_day": [12, 15, 10, 14, 20, 11, 16, 18, 15, 10],
        "crop": [
            "apple",
            "app.",
            "pot.",
            "onion",
            " APPLE ",
            "potato",
            "oni.",
            "Apple",
            "app.",
            "pot.",
        ],
        "chamber": [
            "Chamber 1",
            "Chamber 2",
            "Chamber 3",
            "Chamber 1",
            "Chamber 2",
            "Chamber 3",
            "Chamber 1",
            "Chamber 2",
            "Chamber 2",
            "Chamber 3",
        ],
        "spoilage_found_at_unloading": [4.1, 5.2, 3.0, 8.1, 6.0, 4.5, 7.2, 5.8, 5.2, 3.0],
        "spoilage_pct": [4.2, 5.0, 3.1, 8.0, 6.1, 4.4, 7.0, 5.9, 5.0, 3.1],
    }
    df = pd.DataFrame(data)
    # Explicitly ensure duplicate row exists
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)
    return df


def test_data_cleaning_removes_duplicates_and_leakage(sample_raw_dataframe):
    """
    Test that clean_data properly removes duplicates, strips/normalizes crop values,
    and drops leakage/identifier columns.
    """
    initial_len = len(sample_raw_dataframe)
    cleaned = clean_data(sample_raw_dataframe, drop_leakage=True, drop_identifiers=True, is_train=True)

    # 1. Duplicates should be removed
    assert len(cleaned) < initial_len
    assert cleaned.duplicated().sum() == 0

    # 2. Leakage and identifier columns must be removed
    assert "spoilage_found_at_unloading" not in cleaned.columns
    assert "record_id" not in cleaned.columns
    assert "date" not in cleaned.columns

    # 3. Crop values must be standardized to true levels
    valid_crops = {"apple", "potato", "onion"}
    assert set(cleaned["crop"].unique()).issubset(valid_crops)


def test_pipeline_construction_and_training():
    """
    Test that the Pipeline can be built, trained on synthetic data,
    and produces finite numeric predictions.
    """
    numeric_cols = [
        "storage_days", "temp_set_c", "humidity_pct",
        "power_cuts_hours", "bag_weight_kg", "door_openings_per_day"
    ]
    categorical_cols = ["crop", "chamber"]

    X = pd.DataFrame({
        "storage_days": [20, 40, 60, 30, 50, 70],
        "temp_set_c": [2.0, 3.0, 4.0, 2.5, 3.5, 4.0],
        "humidity_pct": [85.0, 90.0, 80.0, 88.0, 85.0, 92.0],
        "power_cuts_hours": [5.0, 10.0, 2.0, 8.0, 4.0, 6.0],
        "bag_weight_kg": [50.0, 45.0, 55.0, 50.0, 48.0, 52.0],
        "door_openings_per_day": [12, 15, 10, 14, 13, 16],
        "crop": ["apple", "potato", "onion", "apple", "potato", "onion"],
        "chamber": ["Chamber 1", "Chamber 2", "Chamber 3", "Chamber 1", "Chamber 2", "Chamber 3"],
    })
    y = pd.Series([3.5, 6.2, 4.1, 5.0, 7.3, 4.8])

    pipeline = build_pipeline(
        model_type="gradient_boosting",
        hyperparams={"n_estimators": 10, "max_depth": 2, "random_state": 42},
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    fitted_pipe, val_mae, train_mae = train_and_evaluate_model(
        pipeline, X_train=X.iloc[:4], y_train=y.iloc[:4], X_val=X.iloc[4:], y_val=y.iloc[4:]
    )

    preds = fitted_pipe.predict(X.iloc[4:])
    assert len(preds) == 2
    assert not np.isnan(preds).any()
    assert val_mae >= 0.0


def test_mlflow_tracking_and_runs():
    """
    Test that mlflow.db records at least 3 runs, each run contains 'data_sha256'
    and a metric containing 'mae'.
    """
    db_path = Path("mlflow.db")
    if not db_path.is_file():
        pytest.skip("mlflow.db has not been created yet (run training pipeline first).")

    client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
    experiment = client.get_experiment_by_name("cold-storage-spoilage")
    assert experiment is not None, "Experiment 'cold-storage-spoilage' not found in MLflow."

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 3, f"Expected at least 3 MLflow runs, found {len(runs)}"

    for r in runs:
        # Every run must log data_sha256
        assert "data_sha256" in r.data.params, f"Run {r.info.run_id} missing 'data_sha256' param"
        assert len(r.data.params["data_sha256"]) == 64, "data_sha256 is not a valid 64-char hash"

        # Every run must log an MAE metric
        mae_keys = [k for k in r.data.metrics.keys() if "mae" in k.lower()]
        assert len(mae_keys) > 0, f"Run {r.info.run_id} missing an MAE metric"


def test_mlflow_model_registry_and_champion():
    """
    Test that the registered model exists and the 'champion' alias points to a valid model version.
    """
    db_path = Path("mlflow.db")
    if not db_path.is_file():
        pytest.skip("mlflow.db has not been created yet.")

    client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
    model_name = "cold-storage-spoilage-regressor"

    # Verify registered model exists
    model_version = client.get_model_version_by_alias(name=model_name, alias="champion")
    assert model_version is not None, f"Alias '@champion' not found on model '{model_name}'"
    assert model_version.current_stage or model_version.version, "Model version invalid"

    # Verify model file also exists at models/model.joblib
    joblib_model_path = Path("models/model.joblib")
    assert joblib_model_path.is_file(), "models/model.joblib file does not exist."

    loaded_model = joblib.load(joblib_model_path)
    # Test inference on a single row
    sample_row = pd.DataFrame([{
        "storage_days": 45,
        "temp_set_c": 3.0,
        "humidity_pct": 88.0,
        "power_cuts_hours": 12.0,
        "bag_weight_kg": 50.0,
        "door_openings_per_day": 15,
        "crop": "potato",
        "chamber": "Chamber 2",
    }])
    pred = loaded_model.predict(sample_row)
    assert len(pred) == 1
    assert float(pred[0]) > 0.0
