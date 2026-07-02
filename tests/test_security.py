"""
test_security.py
-------------------
Test every cybersecurity control: API key auth, input validation,
abnormal value detection, spoof/replay detection, SHA-256 integrity,
audit logging and the 5 simulated attack scenarios.
"""

import pandas as pd
import pytest

from src.security.security import (
    AbnormalValueDetector, ApiKeyAuthenticator, AuditLogger,
    FileIntegrityChecker, InputValidator, SecurityGateway, SpoofedDataDetector,
)
from src.security.simulated_attacks import run_all_attacks
from src.utils import config

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
}


def test_api_key_authenticator_accepts_valid_key():
    auth = ApiKeyAuthenticator()
    assert auth.is_valid("its-auckland-demo-key-2026") is True


def test_api_key_authenticator_rejects_missing_or_wrong_key():
    auth = ApiKeyAuthenticator()
    assert auth.is_valid(None) is False
    assert auth.is_valid("wrong-key") is False


def test_input_validator_rejects_out_of_range_vehicle_count():
    validator = InputValidator()
    bad_payload = dict(VALID_PAYLOAD, vehicle_count=99999)
    is_valid, error = validator.validate(bad_payload)
    assert is_valid is False
    assert error is not None


def test_input_validator_accepts_good_payload():
    validator = InputValidator()
    is_valid, error = validator.validate(VALID_PAYLOAD)
    assert is_valid is True
    assert error is None


def test_abnormal_value_detector_flags_implausible_combo():
    detector = AbnormalValueDetector()
    problems = detector.check({"avg_speed": 140, "vehicle_count": 300, "queue_length": 0})
    assert len(problems) > 0


def test_spoofed_data_detector_flags_duplicate_timestamp():
    detector = SpoofedDataDetector()
    assert detector.is_duplicate("INT_01", "2026-01-01T08:00:00") is False
    assert detector.is_duplicate("INT_01", "2026-01-01T08:00:00") is True


def test_file_integrity_checker_detects_tampering(tmp_path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text("a,b,c\n1,2,3\n")
    checker = FileIntegrityChecker()
    hash_path = tmp_path / "sample.sha256"
    digest = checker.generate_hash_file(file_path, hash_path)
    assert checker.verify(file_path, hash_path) is True

    # tamper the file - hash must no longer match
    file_path.write_text("a,b,c\n9,9,9\n")
    assert checker.verify(file_path, hash_path) is False
    assert digest != checker.compute_hash(file_path)


def test_audit_logger_appends_event(tmp_path):
    log_path = tmp_path / "audit_log.csv"
    logger = AuditLogger(log_path)
    logger.log_request("INT_01", {"vehicle_count": 10})
    df = pd.read_csv(log_path)
    assert len(df) == 1
    assert df.iloc[0]["event_type"] == "REQUEST"


def test_security_gateway_blocks_missing_api_key():
    gateway = SecurityGateway()
    allowed, reason = gateway.check_request(None, VALID_PAYLOAD)
    assert allowed is False
    assert "Unauthorized" in reason


def test_security_gateway_allows_valid_request():
    gateway = SecurityGateway()
    allowed, reason = gateway.check_request("its-auckland-demo-key-2026", dict(VALID_PAYLOAD, timestamp="2026-05-01T00:00:00"))
    assert allowed is True
    assert reason is None


@pytest.mark.parametrize("result", run_all_attacks())
def test_simulated_attack_is_correctly_blocked(result):
    assert result["passed"] is True, f"Attack '{result['attack_name']}' was NOT blocked"
