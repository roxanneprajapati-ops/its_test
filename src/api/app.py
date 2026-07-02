"""
app.py (API)
------------
Flask REST API - the "Communication Layer" of the ITS architecture.
Simulate the V2I (Vehicle-to-Infrastructure) link: a roadside sensor
or connected vehicle POSTs a JSON reading, the backend run it through
security checks, predict congestion with the trained model, and return
a signal recommendation.

Endpoint:
    GET  /health     -> liveness check, no auth needed
    POST /predict     -> predict congestion level only
    POST /signal       -> predict + adaptive signal recommendation
    GET  /metrics       -> last trained-model metrics report
    GET  /security-log  -> tail of audit log (demo / marker convenience)

Run with:
    python src/api/app.py
"""

from __future__ import annotations

import json
import time

import pandas as pd
from flask import Flask, jsonify, request

from src.models.classification_models import (
    CLASSIFICATION_FEATURES, RandomForestCongestionClassifier,
)
from src.optimization.optimizer import SignalOptimizationEngine
from src.security.security import SecurityGateway
from src.utils import config
from src.utils.logger import log

app = Flask(__name__)

gateway = SecurityGateway()
optimizer_engine = SignalOptimizationEngine()
_classifier: RandomForestCongestionClassifier | None = None


def get_classifier() -> RandomForestCongestionClassifier:
    """Lazy-load the trained model exactly once per process."""
    global _classifier
    if _classifier is None:
        _classifier = RandomForestCongestionClassifier()
        _classifier.load()
    return _classifier


def _build_feature_row(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "Vehicle_Count": payload["vehicle_count"],
        "Avg_Speed": payload["avg_speed"],
        "Occupancy_Pct": payload.get("occupancy_pct", 0.0),
        "Hour": payload["hour"],
        "IsPeakHour": payload["is_peak_hour"],
        "IsWeekend": payload["is_weekend"],
        "Weather_Severity": payload.get("weather_severity", 0),
    }])[CLASSIFICATION_FEATURES]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ITS Adaptive Traffic Signal API"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    api_key = request.headers.get("X-API-Key")

    allowed, reason = gateway.check_request(api_key, payload)
    if not allowed:
        status = 401 if "Unauthorized" in (reason or "") else 400
        return jsonify({"error": reason}), status

    start = time.perf_counter()
    classifier = get_classifier()
    congestion_level = classifier.predict(_build_feature_row(payload))[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = {
        "intersection": payload["intersection_id"],
        "congestion": congestion_level,
        "response_time_ms": round(elapsed_ms, 2),
    }
    gateway.audit_logger.log_prediction(payload["intersection_id"], result)
    return jsonify(result)


@app.route("/signal", methods=["POST"])
def signal():
    payload = request.get_json(silent=True) or {}
    api_key = request.headers.get("X-API-Key")

    allowed, reason = gateway.check_request(api_key, payload)
    if not allowed:
        status = 401 if "Unauthorized" in (reason or "") else 400
        return jsonify({"error": reason}), status

    classifier = get_classifier()
    congestion_level = classifier.predict(_build_feature_row(payload))[0]
    recommendation = optimizer_engine.recommend(
        intersection_id=payload["intersection_id"],
        congestion_level=congestion_level,
        queue_length=payload.get("queue_length", 0),
        incident_detected=False,
    )

    result = recommendation.to_dict()
    gateway.audit_logger.log_prediction(payload["intersection_id"], result)
    return jsonify(result)


@app.route("/metrics", methods=["GET"])
def metrics():
    metrics_path = config.OUTPUT_DIR / "metrics.json"
    if not metrics_path.exists():
        return jsonify({"error": "No metrics found. Run `python main.py train` first."}), 404
    with open(metrics_path) as f:
        return jsonify(json.load(f))


@app.route("/security-log", methods=["GET"])
def security_log():
    if not config.AUDIT_LOG_PATH.exists():
        return jsonify({"events": []})
    df = pd.read_csv(config.AUDIT_LOG_PATH).tail(50)
    return jsonify({"events": df.to_dict(orient="records")})


if __name__ == "__main__":
    log("API", "Starting Flask server on http://127.0.0.1:5000 ...")
    app.run(debug=True, port=5000)
