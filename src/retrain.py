"""
Main CLI execution script for Stage 4: Drift Detection, Retraining Decision,
Model Promotion, and Holdout Predictions.

Run this script using:
    python src/retrain.py
"""

import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from src.spoilage.retrain import run_stage4_pipeline

if __name__ == "__main__":
    print("=" * 70)
    print("Starting Cold Storage Spoilage Stage 4 Pipeline")
    print("Drift Monitoring | Automated Retraining | Champion Promotion | Holdout")
    print("=" * 70)

    results = run_stage4_pipeline("params.yaml")

    print("\n" + "=" * 70)
    print("Stage 4 Pipeline Completed Successfully")
    print("=" * 70)
    print(f"Overall Drift Detected:   {results['drift_results']['overall_drift_detected']}")
    print(f"Most Drifted Feature:     {results['drift_results']['most_drifted_feature']}")
    print(f"Chamber MAE Breakdown:    {results['mae_by_chamber']}")
    print(f"Worst Served Chamber:     {results['worst_served_group']}")
    print(f"v1 MAE on Check (30%):    {results['mae_v1_on_check']:.4f}")
    print(f"v2 MAE on Check (30%):    {results['mae_v2_on_check']:.4f}")
    print(f"Improvement:              {results['improvement_pct'] * 100:.2f}%")
    print(f"Retraining Decision:      {results['decision']}")
    if results['candidate_run_id']:
        print(f"Candidate Run ID:         {results['candidate_run_id']}")
        print(f"Candidate Registered Ver: {results['candidate_version']}")
    print(f"Active Champion Version:  {results['champion_version']}")
    print(f"Holdout Records Scored:   {results['holdout_samples']}")
    print("=" * 70)
