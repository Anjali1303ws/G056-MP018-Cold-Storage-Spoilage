
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_endpoint():
    payload = {
        "storage_days": 30,
        "temp_set_c": 4,
        "humidity_pct": 85,
        "power_cuts_hours": 5,
        "bag_weight_kg": 50,
        "door_openings_per_day": 10,
        "crop": "apple",
        "chamber": "Chamber 1"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "spoilage_pct" in response.json()
    assert isinstance(response.json()["spoilage_pct"], float)


def test_prediction_rejects_invalid_power_cuts():
    payload = {
        "storage_days": 30,
        "temp_set_c": 4,
        "humidity_pct": 85,
        "power_cuts_hours": 160,
        "bag_weight_kg": 50,
        "door_openings_per_day": 10,
        "crop": "apple",
        "chamber": "Chamber 1"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

