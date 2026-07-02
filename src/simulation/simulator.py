"""
simulator.py
------------
This is the "Sensing Layer" of our ITS architecture. Real loop-detector
and camera sensor data from Auckland Transport is not public at 15-min
resolution, so we build realistic synthetic data instead, clearly label
as simulate, follow same schema style as real ITS sensor feed (vehicle
count, speed, occupancy, queue length, weather, incident flag).

Class design (OOP):
    TrafficDataGenerator (abstract base)         -> contract every
                                                     generator must obey
        IntersectionSimulator (one intersection)  -> concrete strategy
    TrafficSimulationManager                      -> coordinate many
                                                     IntersectionSimulator,
                                                     merge to one dataset
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils import config
from src.utils.logger import log


@dataclass
class TrafficRecord:
    """One 15-minute sensor reading at one intersection. Dataclass keep
    field explicit and typed, better than loose dict (good OOP)."""

    timestamp: dt.datetime
    intersection_id: str
    vehicle_count: int
    avg_speed: float
    queue_length: int
    occupancy_pct: float
    delay_seconds: float
    weather: str
    incident_flag: int
    lane_count: int
    time_of_day: str
    day_of_week: str

    def to_dict(self) -> dict:
        return {
            "Timestamp": self.timestamp,
            "Intersection_ID": self.intersection_id,
            "Vehicle_Count": self.vehicle_count,
            "Avg_Speed": round(self.avg_speed, 2),
            "Queue_Length": self.queue_length,
            "Occupancy_Pct": round(self.occupancy_pct, 2),
            "Delay_Seconds": round(self.delay_seconds, 2),
            "Weather": self.weather,
            "Incident_Flag": self.incident_flag,
            "Lane_Count": self.lane_count,
            "Time_of_Day": self.time_of_day,
            "Day_of_Week": self.day_of_week,
        }


class TrafficDataGenerator(ABC):
    """Abstract base. Every concrete generator must implement
    generate(). This demonstrate abstraction / polymorphism for LO3."""

    def __init__(self, seed: int = config.RANDOM_SEED):
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(self) -> list[TrafficRecord]:
        raise NotImplementedError


class IntersectionSimulator(TrafficDataGenerator):
    """Simulate many day of 15-min traffic data for ONE intersection.
    Each intersection has own base volume, peak multiplier and lane
    count so behaviour differ like real junction (CBD vs suburb)."""

    def __init__(self, intersection_id: str, base_volume: int,
                 peak_multiplier: float, lane_count: int,
                 seed: int = config.RANDOM_SEED):
        super().__init__(seed)
        self.intersection_id = intersection_id
        self.base_volume = base_volume
        self.peak_multiplier = peak_multiplier
        self.lane_count = lane_count

    # -- private helper, each do one small job (single responsibility) --
    def _time_of_day_label(self, hour: int) -> str:
        if 5 <= hour < 12:
            return "Morning"
        if 12 <= hour < 17:
            return "Noon"
        if 17 <= hour < 21:
            return "Evening"
        return "Night"

    def _is_peak_hour(self, hour: int) -> bool:
        return (7 <= hour <= 9) or (16 <= hour <= 18)

    def _pick_weather(self) -> str:
        return str(self.rng.choice(config.WEATHER_CONDITIONS, p=config.WEATHER_WEIGHTS))

    def _simulate_vehicle_count(self, hour: int) -> int:
        multiplier = self.peak_multiplier if self._is_peak_hour(hour) else 1.0
        noise = self.rng.normal(0, self.base_volume * 0.08)
        count = self.base_volume * multiplier + noise
        return max(0, int(round(count)))

    def _simulate_speed(self, vehicle_count: int, weather: str) -> float:
        free_flow_speed = 50.0
        congestion_penalty = min(35.0, vehicle_count / 6.0)
        weather_penalty = config.WEATHER_SPEED_PENALTY[weather]
        noise = self.rng.normal(0, 2.0)
        return max(5.0, free_flow_speed - congestion_penalty - weather_penalty + noise)

    def _simulate_queue_length(self, vehicle_count: int, avg_speed: float) -> int:
        speed_factor = max(0.1, (50.0 - avg_speed) / 50.0)
        queue = (vehicle_count / 8.0) * speed_factor
        noise = self.rng.normal(0, 0.35)
        return max(0, int(round(queue + noise)))

    def _simulate_occupancy(self, vehicle_count: int) -> float:
        """% of time the detector loop is occupied by a vehicle. Bound
        0-100, scale with vehicle count and lane count (more lane share
        the load, so occupancy per lane goes down)."""
        raw = (vehicle_count / (self.lane_count * 12.0)) * 100.0
        noise = self.rng.normal(0, 3.0)
        return float(np.clip(raw + noise, 0.0, 100.0))

    def _simulate_delay(self, queue_length: int, avg_speed: float) -> float:
        """Approx delay (second) a vehicle wait at signal, derived from
        queue length and speed - longer queue + slow speed = more delay."""
        delay = queue_length * 2.2 + max(0.0, (50.0 - avg_speed) * 0.6)
        noise = self.rng.normal(0, 2.0)
        return max(0.0, delay + noise)

    def _simulate_incident(self) -> int:
        return int(self.rng.random() < config.INCIDENT_BASE_PROBABILITY)

    # -- main generator ----------------------------------------------
    def generate(self) -> list[TrafficRecord]:
        records: list[TrafficRecord] = []
        start_date = dt.datetime(2026, 1, 1)

        for day in range(config.SIMULATION_DAYS):
            current_date = start_date + dt.timedelta(days=day)
            day_of_week = current_date.strftime("%A")

            for interval in range(config.RECORDS_PER_DAY):
                timestamp = current_date + dt.timedelta(minutes=interval * 15)
                hour = timestamp.hour

                weather = self._pick_weather()
                vehicle_count = self._simulate_vehicle_count(hour)
                avg_speed = self._simulate_speed(vehicle_count, weather)
                queue_length = self._simulate_queue_length(vehicle_count, avg_speed)
                occupancy_pct = self._simulate_occupancy(vehicle_count)
                incident_flag = self._simulate_incident()

                if incident_flag:
                    # incident make speed drop and queue build up, but we
                    # keep the effect modest + noisy on purpose so the
                    # signal overlaps with normal heavy-congestion record.
                    # A model that perfectly separates incident from
                    # normal traffic would be unrealistic - real ITS
                    # incident detection is a genuinely hard, imbalanced
                    # problem, which is exactly why the recall-tuning fix
                    # in src/models/incident_detector.py matters.
                    avg_speed *= self.rng.uniform(0.55, 0.85)
                    queue_length = int(queue_length * self.rng.uniform(1.1, 1.5)) + int(self.rng.integers(0, 3))
                    occupancy_pct = min(100.0, occupancy_pct * self.rng.uniform(1.05, 1.2))

                delay_seconds = self._simulate_delay(queue_length, avg_speed)

                record = TrafficRecord(
                    timestamp=timestamp,
                    intersection_id=self.intersection_id,
                    vehicle_count=vehicle_count,
                    avg_speed=avg_speed,
                    queue_length=queue_length,
                    occupancy_pct=occupancy_pct,
                    delay_seconds=delay_seconds,
                    weather=weather,
                    incident_flag=incident_flag,
                    lane_count=self.lane_count,
                    time_of_day=self._time_of_day_label(hour),
                    day_of_week=day_of_week,
                )
                records.append(record)
        return records


class TrafficSimulationManager:
    """Coordinate multiple IntersectionSimulator object and merge their
    output into one dataset (composition over inheritance pattern)."""

    def __init__(self, profiles: dict | None = None):
        self.profiles = profiles or config.INTERSECTION_PROFILES
        self.simulators: list[IntersectionSimulator] = [
            IntersectionSimulator(
                intersection_id=iid,
                base_volume=vol,
                peak_multiplier=mult,
                lane_count=lanes,
                seed=config.RANDOM_SEED + idx,
            )
            for idx, (iid, (vol, mult, lanes)) in enumerate(self.profiles.items())
        ]

    def run(self) -> pd.DataFrame:
        all_records = []
        for sim in self.simulators:
            all_records.extend(sim.generate())

        df = pd.DataFrame([r.to_dict() for r in all_records])
        df = df.sort_values(["Timestamp", "Intersection_ID"]).reset_index(drop=True)
        return df

    def save(self, path=None) -> pd.DataFrame:
        path = path or config.RAW_DATASET_PATH
        df = self.run()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        log("Simulator", f"Saved {len(df)} records ({len(self.simulators)} intersections) to {path}")
        return df


if __name__ == "__main__":
    manager = TrafficSimulationManager()
    dataset = manager.save()
    print(dataset.head())
