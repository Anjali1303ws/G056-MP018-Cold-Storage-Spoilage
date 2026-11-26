
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------
# 1. Load the trained model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 2. Configure JSON logging
# --------------------------------------------------

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "level": record.levelname,
            "message": record.getMessage(),
        }

        if hasattr(record, "spoilage_pct"):
            log_data["spoilage_pct"] = record.spoilage_pct

        return json.dumps(log_data)


logger = logging.getLogger("mp018_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

logger.propagate = False


# --------------------------------------------------
# 3. Define the prediction request schema
# --------------------------------------------------

class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    storage_days: float
    temp_set_c: float
    humidity_pct: float
    power_cuts_hours: float = Field(le=150)
    bag_weight_kg: float = Field(ge=20)
    door_openings_per_day: float
    crop: str
    chamber: str


# --------------------------------------------------
# 4. Create the FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Cold Storage Spoilage Prediction API",
    description="Predicts spoilage percentage for stored crops.",
    version="1.0",
)


# --------------------------------------------------
# 5. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# 6. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    # Convert the validated request into a DataFrame
    input_data = pd.DataFrame([request.model_dump()])

    # Get prediction from the trained model
    prediction = float(model.predict(input_data)[0])

    # Print one JSON log line for this prediction
    logger.info(
        "prediction_completed",
        extra={"spoilage_pct": prediction},
    )

    return {
        "spoilage_pct": prediction
    }
