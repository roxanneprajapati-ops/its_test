"""
test_api.py
-------------
Test the Flask API endpoints using Flask's test client (no real server
needed). Covers health check, auth rejection, validation rejection and
a successful prediction once a model has been trained.
"""

import pytest

from src.api.app import app as flask_app
from src.models.classification_models import RandomForestCongestionClassifier
from src.preprocessing.preprocessing import PreprocessingPipeline
from src.simulation.simulator import TrafficSimulationManager

VALID_KEY = "its-auckland-demo-key-2026"

VALID_PAYLOAD = {
    "intersection_id": "INT_01",
    "vehicle_count": 40,
    "avg_speed": 35.0,
    "queue_length": 5,
    "occupancy_pct": 30.0,
    "delay_seconds": 10.0,
    "weather_severity": 0,
    "hour": 8,
    "is_peak_hour": 1,
    "is_weekend": 0,
    "timestamp": "2026-06-01T08:00:00",
}


@pytest.fixture(scope="module")
def client():
    flask_app.config.update(TESTING=True)
    return flask_app.test_client()


@pytest.fixture(scope="module", autouse=True)
def ensure_model_trained():
    """The /predict and /signal endpoint need a saved model on disk.
    Train and save one small model before this test module run, so
    test stay independent of whether main.py was run first."""
    raw = TrafficSimulationManager().run()
    processed = PreprocessingPipeline().run(raw)
    clf = RandomForestCongestionClassifier()
    clf.train(processed)
    clf.save()


def test_health_endpoint_does_not_need_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_predict_rejects_missing_api_key(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 401


def test_predict_rejects_invalid_payload(client):
    bad_payload = dict(VALID_PAYLOAD, vehicle_count=99999)
    resp = client.post("/predict", json=bad_payload, headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 400


def test_predict_returns_congestion_with_valid_request(client):
    resp = client.post("/predict", json=VALID_PAYLOAD, headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["congestion"] in ("Low", "Medium", "High")
    assert "response_time_ms" in body


def test_signal_endpoint_returns_recommendation(client):
    payload = dict(VALID_PAYLOAD, timestamp="2026-06-01T08:15:00")
    resp = client.post("/signal", json=payload, headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 200
    body = resp.get_json()
    assert "adaptive_green_seconds" in body
