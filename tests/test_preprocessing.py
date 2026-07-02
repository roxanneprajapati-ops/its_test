"""
test_preprocessing.py
-----------------------
Test DataCleaner, FeatureEngineer and PreprocessingPipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.preprocessing import (
    DataCleaner, DataValidationError, FeatureEngineer, PreprocessingPipeline,
)
from src.simulation.simulator import TrafficSimulationManager


@pytest.fixture(scope="module")
def raw_df():
    return TrafficSimulationManager().run()


def test_cleaner_rejects_missing_required_column():
    cleaner = DataCleaner()
    bad_df = pd.DataFrame({"Timestamp": ["2026-01-01"], "Intersection_ID": ["INT_01"]})
    with pytest.raises(DataValidationError):
        cleaner.clean(bad_df)


def test_cleaner_fills_missing_numeric_values(raw_df):
    df = raw_df.copy()
    df.loc[0, "Vehicle_Count"] = np.nan
    cleaner = DataCleaner()
    cleaned = cleaner.clean(df)
    assert cleaned["Vehicle_Count"].isna().sum() == 0


def test_cleaner_clips_impossible_speed(raw_df):
    df = raw_df.copy()
    df.loc[0, "Avg_Speed"] = 999.0
    cleaned = DataCleaner().clean(df)
    assert cleaned["Avg_Speed"].max() <= 150.0


def test_feature_engineer_adds_time_features(raw_df):
    df = DataCleaner().clean(raw_df)
    df = FeatureEngineer().add_time_features(df)
    for col in ["Hour", "Minute", "DayOfWeekNum", "IsWeekend", "IsPeakHour"]:
        assert col in df.columns


def test_feature_engineer_adds_congestion_label(raw_df):
    df = DataCleaner().clean(raw_df)
    df = FeatureEngineer().add_congestion_label(df)
    assert set(df["Congestion_Level"].unique()).issubset({"Low", "Medium", "High"})


def test_full_pipeline_runs_end_to_end(raw_df):
    processed = PreprocessingPipeline().run(raw_df)
    assert "Congestion_Level" in processed.columns
    assert "Rule_Based_Anomaly_Score" in processed.columns
    assert processed.isna().sum().sum() == 0
