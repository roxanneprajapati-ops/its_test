"""
test_optimizer.py
--------------------
Test the SignalOptimizationEngine: safety constraint, queue adjustment,
incident priority and emergency override.
"""

from src.optimization.optimizer import SignalOptimizationEngine
from src.utils import config


def test_recommendation_never_below_min_green():
    engine = SignalOptimizationEngine()
    rec = engine.recommend("INT_01", "Low", queue_length=0)
    assert rec.recommended_green_time >= config.MIN_GREEN_TIME


def test_recommendation_never_above_max_green():
    engine = SignalOptimizationEngine()
    rec = engine.recommend("INT_01", "High", queue_length=500)
    assert rec.recommended_green_time <= config.MAX_GREEN_TIME


def test_higher_congestion_gives_longer_green():
    engine = SignalOptimizationEngine()
    low = engine.recommend("INT_01", "Low", queue_length=0)
    high = engine.recommend("INT_01", "High", queue_length=0)
    assert high.recommended_green_time > low.recommended_green_time


def test_queue_adjustment_increases_green_time():
    engine = SignalOptimizationEngine()
    small_queue = engine.recommend("INT_01", "Medium", queue_length=2)
    big_queue = engine.recommend("INT_01", "Medium", queue_length=20)
    assert big_queue.recommended_green_time >= small_queue.recommended_green_time


def test_incident_detected_triggers_priority_green():
    engine = SignalOptimizationEngine()
    rec = engine.recommend("INT_01", "Low", queue_length=0, incident_detected=True)
    assert rec.incident_detected is True
    assert "Incident" in rec.reason


def test_emergency_override_returns_short_green():
    engine = SignalOptimizationEngine()
    rec = engine.recommend("INT_01", "High", queue_length=10, emergency_vehicle_detected=True)
    assert rec.recommended_green_time == config.EMERGENCY_OVERRIDE_GREEN_TIME
    assert "Emergency" in rec.reason


def test_recommend_batch_returns_correct_count():
    engine = SignalOptimizationEngine()
    rows = [
        {"intersection_id": "INT_01", "congestion_level": "Low"},
        {"intersection_id": "INT_02", "congestion_level": "High", "queue_length": 12},
    ]
    results = engine.recommend_batch(rows)
    assert len(results) == 2
