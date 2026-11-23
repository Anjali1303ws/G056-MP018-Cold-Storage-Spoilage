"""
Main CLI execution script for training the Cold Storage Spoilage Prediction models.
Run this script using:
    python src/train.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.spoilage.train import run_training_pipeline

if __name__ == "__main__":
    print("Starting Cold Storage Spoilage Model Training Pipeline...")
    results = run_training_pipeline("params.yaml")
    print("\n--- Pipeline Completed Successfully ---")
    print(f"MLflow Experiment: {results['experiment_name']}")
    print(f"Best Model:        {results['best_model_name']}")
    print(f"Best Validation MAE: {results['best_val_mae']:.4f}")
    print(f"Best Run ID:       {results['best_run_id']}")
    print(f"Registered Name:   {results['registered_model_name']}")
    print(f"Champion Version:  {results['champion_version']}")
