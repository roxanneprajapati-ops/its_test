"""
classification_models.py
-------------------------
Congestion-level classifier (Low / Medium / High) - multi-class target.
Compare RandomForestClassifier vs GradientBoostingClassifier as
required by rubric item C. Both report accuracy, precision, recall, F1,
confusion matrix and feature importance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from src.models.base_model import BaseModel
from src.utils import config

CLASSIFICATION_FEATURES = [
    "Vehicle_Count", "Avg_Speed", "Occupancy_Pct", "Hour",
    "IsPeakHour", "IsWeekend", "Weather_Severity",
]
CLASSIFICATION_TARGET = "Congestion_Level"


class BaseCongestionClassifier(BaseModel):
    """Share train/evaluate logic for congestion classifier family."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(config.CONGESTION_LABELS)

    def train(self, df: pd.DataFrame):
        X = df[CLASSIFICATION_FEATURES]
        y = self.label_encoder.transform(df[CLASSIFICATION_TARGET])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.ML_TEST_SIZE,
            random_state=config.ML_RANDOM_STATE, stratify=y,
        )
        self.model.fit(X_train, y_train)

        cv = StratifiedKFold(config.ML_CV_FOLDS, shuffle=True, random_state=config.ML_RANDOM_STATE)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring="f1_weighted")
        self.cv_f1_mean_ = float(cv_scores.mean())
        self.cv_f1_std_ = float(cv_scores.std())

        self.evaluate(X_test, y_test)
        return X_test, y_test

    def predict(self, df: pd.DataFrame) -> list[str]:
        preds = self.model.predict(df[CLASSIFICATION_FEATURES])
        return list(self.label_encoder.inverse_transform(preds))

    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.model.predict(X_test)
        self.metrics_ = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "labels": list(self.label_encoder.classes_),
            "cv_f1_mean": getattr(self, "cv_f1_mean_", None),
            "cv_f1_std": getattr(self, "cv_f1_std_", None),
        }
        return self.metrics_

    def feature_importance(self) -> dict:
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(CLASSIFICATION_FEATURES, self.model.feature_importances_.tolist()))
        return {}


class RandomForestCongestionClassifier(BaseCongestionClassifier):
    def __init__(self):
        super().__init__("congestion_random_forest_classifier")
        self.model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=config.ML_RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )


class GradientBoostingCongestionClassifier(BaseCongestionClassifier):
    def __init__(self):
        super().__init__("congestion_gradient_boosting_classifier")
        self.model = GradientBoostingClassifier(
            n_estimators=config.GB_N_ESTIMATORS,
            learning_rate=config.GB_LEARNING_RATE,
            random_state=config.ML_RANDOM_STATE,
        )
