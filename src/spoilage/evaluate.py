"""
Evaluation and inference helpers for the Cold Storage Spoilage Prediction model.
"""

from typing import Any, Union
import joblib
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def load_champion_model(
    model_source: str = "models:/cold-storage-spoilage-regressor@champion",
    tracking_uri: str = "sqlite:///mlflow.db",
) -> Any:
    """
    Load the registered champion model either via MLflow Model Registry or local joblib.
    """
    if model_source.endswith(".joblib"):
        return joblib.load(model_source)

    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(model_source)


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute Mean Absolute Error (MAE) for a given model and dataset.
    """
    preds = model.predict(X)
    return float(mean_absolute_error(y, preds))


def predict_spoilage(
    model: Any,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Generate spoilage percentage predictions from a feature DataFrame.
    """
    return model.predict(features)
