"""
test_simulation.py
-------------------
Test the sensing layer: TrafficSimulationManager / IntersectionSimulator.
"""

import pandas as pd

from src.simulation.simulator import IntersectionSimulator, TrafficSimulationManager
from src.utils import config


def test_intersection_simulator_generates_records():
    sim = IntersectionSimulator("INT_TEST", base_volume=40, peak_multiplier=2.0, lane_count=2)
    records = sim.generate()
    expected_count = config.SIMULATION_DAYS * config.RECORDS_PER_DAY
    assert len(records) == expected_count


def test_generated_record_fields_in_valid_range():
    sim = IntersectionSimulator("INT_TEST", base_volume=40, peak_multiplier=2.0, lane_count=2)
    records = sim.generate()
    for r in records[:200]:
        assert r.vehicle_count >= 0
        assert 0 <= r.avg_speed <= 150
        assert r.queue_length >= 0
        assert 0 <= r.occupancy_pct <= 100
        assert r.delay_seconds >= 0
        assert r.incident_flag in (0, 1)
        assert r.weather in config.WEATHER_CONDITIONS


def test_simulation_manager_merges_all_intersections():
    manager = TrafficSimulationManager()
    df = manager.run()
    assert isinstance(df, pd.DataFrame)
    assert set(df["Intersection_ID"].unique()) == set(config.INTERSECTION_IDS)
    assert len(df) == config.SIMULATION_DAYS * config.RECORDS_PER_DAY * len(config.INTERSECTION_IDS)


def test_required_columns_present():
    df = TrafficSimulationManager().run()
    required = ["Timestamp", "Intersection_ID", "Vehicle_Count", "Avg_Speed",
                "Queue_Length", "Occupancy_Pct", "Delay_Seconds", "Weather", "Incident_Flag"]
    for col in required:
        assert col in df.columns
