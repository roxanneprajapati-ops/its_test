"""
optimizer.py
------------
The "Application Layer" decision-support engine. Turn ML prediction
(congestion level, incident flag, queue length) into an actual signal
timing recommendation, and compares it against a FIXED-TIME baseline
signal plan (the thing most older intersections run today).

Rule implement (rubric item E):
  - minimum / maximum green time (safety constraint, signal must never
    go below or above a safe bound)
  - queue-based adjustment (longer queue -> longer green, capped)
  - congestion-based adjustment (Low/Medium/High base timing)
  - incident-aware priority (incident detect -> give that approach a
    long priority green to clear the back-up)
  - emergency override placeholder (a stub method a real ITS system
    would call from an emergency-vehicle pre-emption sensor / V2X
    message - kept simple on purpose since real hardware integration
    is out of scope for this assignment)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils import config


@dataclass
class SignalRecommendation:
    intersection_id: str
    congestion_level: str
    incident_detected: bool
    fixed_time_green: int
    recommended_green_time: int
    reason: str

    def to_dict(self) -> dict:
        return {
            "intersection": self.intersection_id,
            "congestion": self.congestion_level,
            "incident_detected": self.incident_detected,
            "fixed_time_green_seconds": self.fixed_time_green,
            "adaptive_green_seconds": self.recommended_green_time,
            "reason": self.reason,
        }


class SafetyConstraintError(Exception):
    """Raise if an internal calculation ever produce a timing outside
    the safe min/max bound - should never happen but fail loud if it
    does, instead of silently sending an unsafe value downstream."""


class SignalOptimizationEngine:
    """Encapsulate the rule-based logic mapping ML output to a signal
    timing decision. Kept separate from the ML model class so the
    optimisation POLICY can change independently of how congestion is
    predicted (separation of concerns / single responsibility)."""

    def __init__(self, timing_map: dict | None = None,
                 fixed_green_time: int = config.BASE_GREEN_TIME,
                 min_green: int = config.MIN_GREEN_TIME,
                 max_green: int = config.MAX_GREEN_TIME):
        self.timing_map = timing_map or config.SIGNAL_TIMING_MAP
        self.fixed_green_time = fixed_green_time
        self.min_green = min_green
        self.max_green = max_green

    def _clamp(self, seconds: int) -> int:
        """Safety constraint - never let signal go outside safe bound."""
        clamped = max(self.min_green, min(self.max_green, seconds))
        return int(clamped)

    def _queue_adjustment(self, queue_length: int) -> int:
        """Extra second of green per vehicle waiting beyond 5, capped
        at +25s so one huge outlier queue can't break the cycle."""
        extra = max(0, queue_length - 5) * 1.5
        return int(min(25, extra))

    def emergency_override(self, emergency_vehicle_detected: bool) -> int | None:
        """Placeholder hook: in a full ITS deployment this would react
        to a V2X / transponder signal from an approaching emergency
        vehicle and force a short green for the cross street so the
        emergency vehicle's path is cleared immediately. Returns the
        override green time, or None if no override is needed."""
        if emergency_vehicle_detected:
            return config.EMERGENCY_OVERRIDE_GREEN_TIME
        return None

    def recommend(self, intersection_id: str, congestion_level: str,
                   queue_length: int = 0, incident_detected: bool = False,
                   emergency_vehicle_detected: bool = False) -> SignalRecommendation:
        override = self.emergency_override(emergency_vehicle_detected)
        if override is not None:
            safe_override = max(5, min(self.max_green, override))
            return SignalRecommendation(
                intersection_id=intersection_id,
                congestion_level=congestion_level,
                incident_detected=incident_detected,
                fixed_time_green=self.fixed_green_time,
                recommended_green_time=safe_override,
                reason="Emergency vehicle override applied",
            )

        if incident_detected:
            green = self._clamp(config.INCIDENT_PRIORITY_GREEN_TIME)
            reason = "Incident detected: priority green extended to clear back-up"
        else:
            base = self.timing_map.get(congestion_level, self.fixed_green_time)
            adjusted = base + self._queue_adjustment(queue_length)
            green = self._clamp(adjusted)
            reason = (f"Congestion={congestion_level}, queue={queue_length} -> "
                      f"base {base}s + queue adjustment, clamped to safe range "
                      f"[{self.min_green}-{self.max_green}]s")

        return SignalRecommendation(
            intersection_id=intersection_id,
            congestion_level=congestion_level,
            incident_detected=incident_detected,
            fixed_time_green=self.fixed_green_time,
            recommended_green_time=green,
            reason=reason,
        )

    def recommend_batch(self, rows: list[dict]) -> list[SignalRecommendation]:
        """rows: list of dict with keys intersection_id, congestion_level,
        queue_length, incident_detected, emergency_vehicle_detected
        (last two optional)."""
        return [
            self.recommend(
                intersection_id=r["intersection_id"],
                congestion_level=r["congestion_level"],
                queue_length=r.get("queue_length", 0),
                incident_detected=r.get("incident_detected", False),
                emergency_vehicle_detected=r.get("emergency_vehicle_detected", False),
            )
            for r in rows
        ]
