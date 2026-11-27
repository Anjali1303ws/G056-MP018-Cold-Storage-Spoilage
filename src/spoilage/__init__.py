"""
Cold Storage Spoilage Prediction Package (MP018 / Group G056).
"""

from .data import clean_crop, clean_data, compute_data_sha256, get_preprocessor, load_train_data
from .drift import (
    calculate_categorical_drift,
    calculate_ks_test,
    calculate_psi,
    detect_dataset_drift,
    run_drift_detection,
    save_drift_report,
)
from .evaluate import evaluate_model, load_champion_model, predict_spoilage
from .retrain import (
    compute_chamber_mae,
    generate_holdout_predictions,
    run_stage4_pipeline,
    split_later_dataset,
    update_findings_yaml,
)
from .train import build_pipeline, run_training_pipeline, train_and_evaluate_model

__all__ = [
    # Data & Preprocessing
    "clean_data",
    "clean_crop",
    "compute_data_sha256",
    "load_train_data",
    "get_preprocessor",
    # Training & Pipelines
    "build_pipeline",
    "train_and_evaluate_model",
    "run_training_pipeline",
    # Evaluation & Inference
    "load_champion_model",
    "evaluate_model",
    "predict_spoilage",
    # Drift Monitoring
    "calculate_psi",
    "calculate_ks_test",
    "calculate_categorical_drift",
    "detect_dataset_drift",
    "run_drift_detection",
    "save_drift_report",
    # Retraining & Stage 4
    "compute_chamber_mae",
    "split_later_dataset",
    "generate_holdout_predictions",
    "update_findings_yaml",
    "run_stage4_pipeline",
]
