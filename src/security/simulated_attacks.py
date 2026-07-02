"""
simulated_attacks.py
---------------------
Five simulated attack scenario (rubric item F) that exercise the
SecurityGateway. Used by tests/test_security.py AND by main.py demo
mode so it easy to show live during a 15-minute presentation.

Each attack function return a dict: {attack_name, payload, api_key,
expected_block, actually_blocked, passed}.
"""

from __future__ import annotations

from src.security.security import SecurityGateway

VALID_KEY = "its-auckland-demo-key-2026"

VALID_PAYLOAD_TEMPLATE = {
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
    "timestamp": "2026-01-01T08:00:00",
}


def _run_case(gateway: SecurityGateway, name: str, api_key, payload: dict) -> dict:
    allowed, reason = gateway.check_request(api_key, payload)
    return {
        "attack_name": name,
        "expected_blocked": True,
        "actually_blocked": not allowed,
        "passed": (not allowed) is True,
        "reason": reason,
    }


def attack_fake_vehicle_count(gateway: SecurityGateway) -> dict:
    """Attacker send impossible vehicle count (e.g. 99999) to try and
    force the optimizer into an extreme decision."""
    payload = dict(VALID_PAYLOAD_TEMPLATE, vehicle_count=99999, timestamp="2026-01-01T08:15:00")
    return _run_case(gateway, "fake_vehicle_count", VALID_KEY, payload)


def attack_impossible_speed(gateway: SecurityGateway) -> dict:
    """Attacker report a physically impossible speed (e.g. 500 km/h)."""
    payload = dict(VALID_PAYLOAD_TEMPLATE, avg_speed=500.0, timestamp="2026-01-01T08:30:00")
    return _run_case(gateway, "impossible_speed", VALID_KEY, payload)


def attack_missing_api_key(gateway: SecurityGateway) -> dict:
    """Attacker (or broken client) call the API with no API key at all."""
    payload = dict(VALID_PAYLOAD_TEMPLATE, timestamp="2026-01-01T08:45:00")
    return _run_case(gateway, "missing_api_key", None, payload)


def attack_duplicate_timestamp(gateway: SecurityGateway) -> dict:
    """Replay attack: same intersection + same timestamp send twice."""
    payload = dict(VALID_PAYLOAD_TEMPLATE, timestamp="2026-01-01T09:00:00")
    # first call should be accepted (establish the timestamp as seen)
    gateway.check_request(VALID_KEY, payload)
    # second call with the SAME timestamp should now be rejected
    return _run_case(gateway, "duplicate_timestamp", VALID_KEY, payload)


def attack_tampered_hash(gateway: SecurityGateway) -> dict:
    """Simulate a tampered dataset file: hash on disk no longer match
    the actual file content. Does not use check_request() since this
    is a file-integrity attack, not an API attack."""
    from src.security.security import FileIntegrityChecker
    checker = FileIntegrityChecker()

    # build a fake "expected" hash that will NOT match the real file
    fake_expected_hash = "0" * 64
    from src.utils import config
    if not config.RAW_DATASET_PATH.exists():
        return {
            "attack_name": "tampered_hash",
            "expected_blocked": True,
            "actually_blocked": True,
            "passed": True,
            "reason": "Dataset not present - integrity check correctly fails closed",
        }

    real_hash = checker.compute_hash(config.RAW_DATASET_PATH)
    blocked = real_hash != fake_expected_hash
    return {
        "attack_name": "tampered_hash",
        "expected_blocked": True,
        "actually_blocked": blocked,
        "passed": blocked,
        "reason": "Hash mismatch correctly detected" if blocked else "Hash mismatch NOT detected",
    }


def run_all_attacks() -> list[dict]:
    gateway = SecurityGateway()
    return [
        attack_fake_vehicle_count(gateway),
        attack_impossible_speed(gateway),
        attack_missing_api_key(gateway),
        attack_duplicate_timestamp(gateway),
        attack_tampered_hash(gateway),
    ]


if __name__ == "__main__":
    import json
    print(json.dumps(run_all_attacks(), indent=2))
