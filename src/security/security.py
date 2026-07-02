"""
security.py
------------
Cybersecurity control layer for the ITS system (rubric item F). An ITS
backend is a juicy attack target - fake sensor data could trick the
optimizer into giving the wrong street a long green light, so every
control here map to a real V2I threat:

  1. ApiKeyAuthenticator     - reject request with no/wrong API key
  2. InputValidator (pydantic) - reject malformed / out-of-range payload
  3. AbnormalValueDetector    - flag physically-impossible sensor value
  4. SpoofedDataDetector      - flag duplicate-timestamp (replay attack)
  5. FileIntegrityChecker     - SHA-256 hash, detect dataset tampering
  6. AuditLogger              - append-only CSV trail of every event
  7. SecurityGateway          - facade that run ALL check in one call,
                                used by api.py before any data reach ML

simulated_attacks.py (separate file) drives these classes through five
attack scenario for the test suite / live demo.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from src.utils import config


# ---------------------------------------------------------------------
# 1. API key authentication
# ---------------------------------------------------------------------
class ApiKeyAuthenticator:
    """Shared-secret API key check. Production system would use OAuth2
    / mTLS, but a static key is enough to demonstrate access control
    for this assignment scope."""

    def __init__(self, valid_keys: set[str] | None = None):
        self.valid_keys = valid_keys or config.VALID_API_KEYS

    def is_valid(self, supplied_key: str | None) -> bool:
        return supplied_key is not None and supplied_key in self.valid_keys


# ---------------------------------------------------------------------
# 2. Input validation (pydantic schema)
# ---------------------------------------------------------------------
class PredictionRequestSchema(BaseModel):
    """Strict schema every incoming prediction request must satisfy
    before it touch the ML model. Range bound chosen from physically
    possible value for an urban intersection."""

    intersection_id: str = Field(..., pattern=r"^INT_0[1-4]$")
    vehicle_count: int = Field(..., ge=0, le=config.MAX_VEHICLE_COUNT)
    avg_speed: float = Field(..., ge=0.0, le=config.MAX_SPEED_KMH)
    queue_length: int = Field(0, ge=0, le=300)
    occupancy_pct: float = Field(0.0, ge=0.0, le=100.0)
    delay_seconds: float = Field(0.0, ge=0.0, le=600.0)
    weather_severity: int = Field(0, ge=0, le=3)
    hour: int = Field(..., ge=0, le=23)
    is_peak_hour: int = Field(..., ge=0, le=1)
    is_weekend: int = Field(..., ge=0, le=1)
    timestamp: str | None = None


class InputValidator:
    """Thin wrapper around the pydantic schema, returns a simple
    (is_valid, error) tuple so caller don't need to know about
    pydantic exception type."""

    def validate(self, payload: dict) -> tuple[bool, str | None]:
        try:
            PredictionRequestSchema(**payload)
            return True, None
        except ValidationError as exc:
            return False, str(exc)


# ---------------------------------------------------------------------
# 3. Abnormal value detection (defence beyond simple range check)
# ---------------------------------------------------------------------
class AbnormalValueDetector:
    """Catch value that pass the basic schema range check but is still
    physically nonsense, e.g. very high speed WITH very high vehicle
    count at same time (impossible in real traffic), or queue length
    that make no sense for the reported vehicle count."""

    def check(self, payload: dict) -> list[str]:
        problems: list[str] = []
        speed = payload.get("avg_speed", 0)
        vehicle_count = payload.get("vehicle_count", 0)
        queue_length = payload.get("queue_length", 0)

        if speed > 120 and vehicle_count > 200:
            problems.append("Implausible: very high speed reported together with very high vehicle count")
        if queue_length > vehicle_count + 50:
            problems.append("Implausible: queue length far exceeds reported vehicle count")
        if speed <= 0 and vehicle_count == 0:
            problems.append("Suspicious: zero speed and zero vehicle is unusual sensor dead-reading")
        return problems


# ---------------------------------------------------------------------
# 4. Spoofed / replay data detection
# ---------------------------------------------------------------------
class SpoofedDataDetector:
    """Detect duplicate-timestamp record per intersection, a classic
    sign of a replay / spoofing attack where an attacker resend an old
    valid-looking sensor packet to confuse the system."""

    def __init__(self):
        self._seen: set[tuple[str, str]] = set()

    def is_duplicate(self, intersection_id: str, timestamp: str) -> bool:
        key = (intersection_id, timestamp)
        if key in self._seen:
            return True
        self._seen.add(key)
        return False

    def reset(self) -> None:
        self._seen.clear()


# ---------------------------------------------------------------------
# 5. File integrity (SHA-256)
# ---------------------------------------------------------------------
class FileIntegrityChecker:
    """Compute / verify SHA-256 hash of the traffic dataset so any
    tampering or corruption is caught before the data is trusted."""

    @staticmethod
    def compute_hash(path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def generate_hash_file(self, data_path: Path = config.RAW_DATASET_PATH,
                            hash_path: Path = config.DATASET_HASH_PATH) -> str:
        digest = self.compute_hash(data_path)
        hash_path.parent.mkdir(parents=True, exist_ok=True)
        hash_path.write_text(digest)
        return digest

    def verify(self, data_path: Path = config.RAW_DATASET_PATH,
               hash_path: Path = config.DATASET_HASH_PATH) -> bool:
        if not hash_path.exists() or not data_path.exists():
            return False
        expected = hash_path.read_text().strip()
        actual = self.compute_hash(data_path)
        return expected == actual


# ---------------------------------------------------------------------
# 6. Audit logging
# ---------------------------------------------------------------------
class AuditLogger:
    """Append-only CSV trail of every request, prediction and security
    event, for traceability and accountability."""

    FIELDNAMES = ["timestamp", "event_type", "intersection_id", "detail"]

    def __init__(self, log_path: Path = config.AUDIT_LOG_PATH):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def log(self, event_type: str, intersection_id: str = "-", detail: str = "") -> None:
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow({
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "event_type": event_type,
                "intersection_id": intersection_id,
                "detail": detail,
            })

    def log_request(self, intersection_id: str, payload: dict) -> None:
        self.log("REQUEST", intersection_id, str(payload))

    def log_prediction(self, intersection_id: str, prediction: dict) -> None:
        self.log("PREDICTION", intersection_id, str(prediction))

    def log_error(self, detail: str, intersection_id: str = "-") -> None:
        self.log("ERROR", intersection_id, detail)

    def log_security_event(self, detail: str, intersection_id: str = "-") -> None:
        self.log("SECURITY_ALERT", intersection_id, detail)


# ---------------------------------------------------------------------
# 7. Security gateway facade - one call runs every check in order
# ---------------------------------------------------------------------
class SecurityGateway:
    """Facade that runs API-key check, schema validation, abnormal
    value check and replay check in one call, and logs every outcome.
    api.py calls ONLY this class, never the individual checker, so the
    security policy stay centralised and consistent."""

    def __init__(self):
        self.authenticator = ApiKeyAuthenticator()
        self.validator = InputValidator()
        self.abnormal_detector = AbnormalValueDetector()
        self.spoof_detector = SpoofedDataDetector()
        self.audit_logger = AuditLogger()

    def check_request(self, api_key: str | None, payload: dict) -> tuple[bool, str | None]:
        """Run the full security pipeline. Return (allowed, reason)."""
        if not self.authenticator.is_valid(api_key):
            self.audit_logger.log_security_event("Missing or invalid API key", payload.get("intersection_id", "-"))
            return False, "Unauthorized: missing or invalid API key"

        is_valid, error = self.validator.validate(payload)
        if not is_valid:
            self.audit_logger.log_security_event(f"Schema validation failed: {error}",
                                                   payload.get("intersection_id", "-"))
            return False, f"Invalid request payload: {error}"

        abnormal = self.abnormal_detector.check(payload)
        if abnormal:
            self.audit_logger.log_security_event(f"Abnormal value detected: {abnormal}",
                                                   payload.get("intersection_id", "-"))
            return False, f"Abnormal sensor value rejected: {abnormal}"

        timestamp = payload.get("timestamp")
        if timestamp and self.spoof_detector.is_duplicate(payload.get("intersection_id", "-"), timestamp):
            self.audit_logger.log_security_event("Duplicate timestamp (possible replay attack)",
                                                   payload.get("intersection_id", "-"))
            return False, "Duplicate timestamp rejected (possible spoofed/replay data)"

        self.audit_logger.log_request(payload.get("intersection_id", "-"), payload)
        return True, None
