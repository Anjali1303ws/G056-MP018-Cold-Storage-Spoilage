"""
Model training, evaluation, MLflow tracking, and model registration.
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import yaml

from .data import compute_data_sha256, get_preprocessor, load_train_data


def build_pipeline(
    model_type: str,
    hyperparams: dict,
    numeric_cols: list,
    categorical_cols: list,
) -> Pipeline:
    """
    Construct a full end-to-end Pipeline with preprocessing and the regressor.
    """
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)

    if model_type == "random_forest":
        regressor = RandomForestRegressor(**hyperparams)
    elif model_type == "gradient_boosting":
        regressor = GradientBoostingRegressor(**hyperparams)
    elif model_type == "extra_trees":
        regressor = ExtraTreesRegressor(**hyperparams)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])


def train_and_evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[Pipeline, float, float]:
    """
    Train a pipeline and compute validation MAE and training MAE.
    """
    pipeline.fit(X_train, y_train)
    train_preds = pipeline.predict(X_train)
    val_preds = pipeline.predict(X_val)

    train_mae = float(mean_absolute_error(y_train, train_preds))
    val_mae = float(mean_absolute_error(y_val, val_preds))

    return pipeline, val_mae, train_mae


def update_findings_tracking(
    findings_path: str,
    experiment_name: str,
    best_run_id: str,
    registered_model_name: str,
    champion_version: int,
) -> None:
    """
    Update the tracking section of findings.yaml while preserving the rest of the file.
    """
    p = Path(findings_path)
    if not p.is_file():
        return

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "tracking" not in data:
        data["tracking"] = {}

    data["tracking"]["experiment_name"] = experiment_name
    data["tracking"]["best_run_id"] = best_run_id
    data["tracking"]["registered_model_name"] = registered_model_name
    data["tracking"]["champion_version"] = champion_version

    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def run_training_pipeline(params_path: str = "params.yaml") -> Dict[str, Any]:
    """
    Execute the Stage 2 training pipeline:
    1. Read configuration from params.yaml.
    2. Compute SHA-256 of original raw train.csv.
    3. Load and clean training data, split into train/val.
    4. Track at least 3 model configurations in MLflow (SQLite).
    5. Register the best model and assign the 'champion' alias.
    6. Save local model to models/model.joblib.
    7. Update reports/findings.yaml.
    """
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # 1. SHA-256 hash of original train.csv
    train_path = params["data"]["raw_train_path"]
    data_sha256 = compute_data_sha256(train_path)

    # 2. Data loading and train/val split
    X, y = load_train_data(train_path, params)
    test_size = params["data"].get("test_size", 0.2)
    random_state = params["data"].get("random_state", 42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    numeric_cols = params["data"]["numeric_cols"]
    categorical_cols = params["data"]["categorical_cols"]

    # 3. Setup MLflow tracking
    tracking_uri = params["mlflow"]["tracking_uri"]
    experiment_name = params["mlflow"]["experiment_name"]
    registered_model_name = params["mlflow"]["registered_model_name"]
    champion_alias = params["mlflow"].get("champion_alias", "champion")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    run_results = []
    trained_pipelines = {}

    # 4. Train and log each model configuration
    models_config = params["models"]
    for model_name, hyperparams in models_config.items():
        pipeline = build_pipeline(model_name, hyperparams, numeric_cols, categorical_cols)

        with mlflow.start_run(run_name=f"{model_name}_v1") as run:
            run_id = run.info.run_id

            # Train and evaluate
            fitted_pipe, val_mae, train_mae = train_and_evaluate_model(
                pipeline, X_train, y_train, X_val, y_val
            )

            # Log mandatory data_sha256 and hyperparameters
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", model_name)
            for param_key, param_val in hyperparams.items():
                mlflow.log_param(param_key, param_val)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))

            # Log MAE metric (mandatory requirement from handbook)
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("train_mae", train_mae)

            # Log pipeline artifact with cloudpickle serialization for safety
            mlflow.sklearn.log_model(
                sk_model=fitted_pipe,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            run_results.append({
                "model_name": model_name,
                "run_id": run_id,
                "val_mae": val_mae,
                "train_mae": train_mae,
            })
            trained_pipelines[run_id] = fitted_pipe
            print(f"Logged {model_name} (Run ID: {run_id}) with Validation MAE: {val_mae:.4f}")

    # 5. Select best model based on validation MAE (lower is better)
    best_result = min(run_results, key=lambda x: x["val_mae"])
    best_run_id = best_result["run_id"]
    best_pipeline = trained_pipelines[best_run_id]
    best_model_name = best_result["model_name"]
    best_val_mae = best_result["val_mae"]

    print(f"\nBest Model: {best_model_name} (Run ID: {best_run_id}) with MAE: {best_val_mae:.4f}")

    # 6. Model Registration and Champion Alias
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )
    version_number = int(model_version.version)

    # Set champion alias
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=champion_alias,
        version=str(version_number),
    )
    print(f"Registered model '{registered_model_name}' version {version_number} with alias '@{champion_alias}'")

    # 7. Save champion model to models/model.joblib
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, models_dir / "model.joblib")
    print(f"Saved champion pipeline to {models_dir / 'model.joblib'}")

    # 8. Update findings.yaml tracking section
    update_findings_tracking(
        findings_path="reports/findings.yaml",
        experiment_name=experiment_name,
        best_run_id=best_run_id,
        registered_model_name=registered_model_name,
        champion_version=version_number,
    )
    print("Updated reports/findings.yaml tracking section.")

    return {
        "experiment_name": experiment_name,
        "runs": run_results,
        "best_run_id": best_run_id,
        "best_model_name": best_model_name,
        "best_val_mae": best_val_mae,
        "registered_model_name": registered_model_name,
        "champion_version": version_number,
        "champion_alias": champion_alias,
        "data_sha256": data_sha256,
    }
