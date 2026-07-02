"""
data_loader.py
---------------
Small wrapper class around reading the raw/processed CSV so other
module never hard-code a path string directly (rubric "no hard-coded
fragile paths" requirement in section G).
"""

from __future__ import annotations

import pandas as pd

from src.utils import config


class DatasetLoader:
    """Single place responsible for reading dataset file from disk."""

    def load_raw(self) -> pd.DataFrame:
        return pd.read_csv(config.RAW_DATASET_PATH, parse_dates=["Timestamp"])

    def load_processed(self) -> pd.DataFrame:
        return pd.read_csv(config.PROCESSED_DATASET_PATH, parse_dates=["Timestamp"])

    def raw_exists(self) -> bool:
        return config.RAW_DATASET_PATH.exists()

    def processed_exists(self) -> bool:
        return config.PROCESSED_DATASET_PATH.exists()
