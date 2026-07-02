"""
base_model.py
-------------
Common base class so every ML model in this project share the same
save()/load() contract (joblib). This is "programming to an interface"
- a basic OOP design principle examiner usually look for.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from src.utils import config


class BaseModel:
    """Shared persistence logic for all model wrapper class below."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.metrics_: dict = {}

    def save(self, directory: Path = config.MODELS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        out_path = directory / f"{self.model_name}.joblib"
        joblib.dump(self.model, out_path)
        return out_path

    def load(self, directory: Path = config.MODELS_DIR) -> "BaseModel":
        self.model = joblib.load(directory / f"{self.model_name}.joblib")
        return self
