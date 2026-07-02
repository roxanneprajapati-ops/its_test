"""
preprocessing.py
-----------------
Clean raw simulate dataset and engineer feature/label need by ML model.
Single pipeline class so api.py and dashboard use exact same transform
that train.py used (avoid train/serve skew - important real-world ML
engineering point for LO3 marking).
"""

from __future__ import annotations

import pandas as pd

from src.utils import config


class DataValidationError(Exception):
    """Raise when raw data fail basic sanity check before it enter
    pipeline. Keep this separate from security module reject-rule,
    this one is about DATA QUALITY not malicious input."""


class DataCleaner:
    """Handle missing value, duplicate row and type coercion."""

    REQUIRED_COLUMNS = [
        "Timestamp", "Intersection_ID", "Vehicle_Count", "Avg_Speed",
        "Queue_Length", "Occupancy_Pct", "Delay_Seconds", "Weather",
        "Incident_Flag",
    ]

    def validate_schema(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataValidationError(f"Missing required column(s): {missing}")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_schema(df)
        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.drop_duplicates(subset=["Timestamp", "Intersection_ID"])

        # missing data handling - fill numeric gap with intersection median
        # instead of dropping whole row, so we don't waste good record
        numeric_cols = ["Vehicle_Count", "Avg_Speed", "Queue_Length",
                         "Occupancy_Pct", "Delay_Seconds"]
        for col in numeric_cols:
            df[col] = df.groupby("Intersection_ID")[col].transform(
                lambda s: s.fillna(s.median())
            )
        df = df.dropna(subset=numeric_cols)

        # clip impossible / sensor-error value (defensive data quality gate)
        df["Vehicle_Count"] = df["Vehicle_Count"].clip(lower=0, upper=config.MAX_VEHICLE_COUNT)
        df["Avg_Speed"] = df["Avg_Speed"].clip(lower=0, upper=config.MAX_SPEED_KMH)
        df["Queue_Length"] = df["Queue_Length"].clip(lower=0)
        df["Occupancy_Pct"] = df["Occupancy_Pct"].clip(lower=0, upper=100)
        df["Delay_Seconds"] = df["Delay_Seconds"].clip(lower=0)
        return df


class FeatureEngineer:
    """Add derive numeric/categorical feature used by ML model."""

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Hour"] = df["Timestamp"].dt.hour
        df["Minute"] = df["Timestamp"].dt.minute
        df["DayOfWeekNum"] = df["Timestamp"].dt.dayofweek
        df["IsWeekend"] = (df["DayOfWeekNum"] >= 5).astype(int)
        df["IsPeakHour"] = df["Hour"].apply(lambda h: 1 if (7 <= h <= 9 or 16 <= h <= 18) else 0)
        return df

    def add_weather_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        weather_severity = {"Clear": 0, "Rain": 1, "Fog": 2, "Heavy_Rain": 3}
        df["Weather_Severity"] = df["Weather"].map(weather_severity).fillna(0).astype(int)
        return df

    def add_congestion_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label Low/Medium/High from queue length, become classifier
        target. Threshold come from config so it easy to tune."""
        df = df.copy()

        def classify(q):
            if q <= config.QUEUE_LOW_MAX:
                return "Low"
            if q <= config.QUEUE_MEDIUM_MAX:
                return "Medium"
            return "High"

        df["Congestion_Level"] = df["Queue_Length"].apply(classify)
        return df

    def add_rule_based_incident_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple rolling-baseline anomaly rule (speed drop + volume
        spike). Kept ONLY as a cheap heuristic feature fed into the ML
        incident classifier, NOT the final detector - the final
        detector is the trained RandomForestClassifier in
        incident_detector.py, which gives much better recall."""
        df = df.copy()
        df = df.sort_values(["Intersection_ID", "Timestamp"])

        speed_baseline = df.groupby("Intersection_ID")["Avg_Speed"].transform(
            lambda s: s.rolling(8, min_periods=1).mean())
        volume_baseline = df.groupby("Intersection_ID")["Vehicle_Count"].transform(
            lambda s: s.rolling(8, min_periods=1).mean())

        speed_drop_pct = (speed_baseline - df["Avg_Speed"]) / speed_baseline.replace(0, 1)
        volume_increase_pct = (df["Vehicle_Count"] - volume_baseline) / volume_baseline.replace(0, 1)

        df["Rule_Based_Anomaly_Score"] = (speed_drop_pct.clip(lower=0) +
                                           volume_increase_pct.clip(lower=0))
        return df.sort_values(["Timestamp", "Intersection_ID"]).reset_index(drop=True)


class PreprocessingPipeline:
    """Orchestrate DataCleaner + FeatureEngineer. Single entry point
    re-used by train.py, api and dashboard so output always same shape."""

    def __init__(self):
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.cleaner.clean(df)
        df = self.engineer.add_time_features(df)
        df = self.engineer.add_weather_encoding(df)
        df = self.engineer.add_congestion_label(df)
        df = self.engineer.add_rule_based_incident_flag(df)
        return df

    def run_from_csv(self, path=None) -> pd.DataFrame:
        path = path or config.RAW_DATASET_PATH
        df = pd.read_csv(path)
        return self.run(df)

    def save(self, df: pd.DataFrame, path=None) -> None:
        path = path or config.PROCESSED_DATASET_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    processed = pipeline.run_from_csv()
    pipeline.save(processed)
    print(processed["Congestion_Level"].value_counts())
    print("Incident rate:", processed["Incident_Flag"].mean())
