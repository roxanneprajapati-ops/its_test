"""
regression_models.py
---------------------
Queue-length prediction (continuous target). Compare THREE regressor as
required by rubric item C:

    1. LinearRegressionModel    - simple baseline
    2. RandomForestRegressorModel
    3. GradientBoostingRegressorModel

Each share .train()/.predict()/.evaluate() interface from BaseModel so
evaluation.py can loop over them generically (polymorphism).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src.models.base_model import BaseModel
from src.utils import config

REGRESSION_FEATURES = [
    "Vehicle_Count", "Avg_Speed", "Occupancy_Pct", "Hour",
    "IsPeakHour", "IsWeekend", "Weather_Severity",
]
REGRESSION_TARGET = "Queue_Length"


class BaseRegressor(BaseModel):
    """Common train/evaluate logic shared by every regressor below.
    Sub-class only need to set self.model in __init__ (Template
    Method pattern)."""

    def train(self, df: pd.DataFrame):
        X = df[REGRESSION_FEATURES]
        y = df[REGRESSION_TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.ML_TEST_SIZE, random_state=config.ML_RANDOM_STATE,
        )
        self.model.fit(X_train, y_train)

        # cross-validation on the training fold - rubric ask for this
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=KFold(config.ML_CV_FOLDS, shuffle=True,
                                                     random_state=config.ML_RANDOM_STATE),
            scoring="neg_mean_absolute_error",
        )
        self.cv_mae_mean_ = float(-cv_scores.mean())
        self.cv_mae_std_ = float(cv_scores.std())

        self.evaluate(X_test, y_test)
        return X_test, y_test

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[REGRESSION_FEATURES])

    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        self.metrics_ = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, y_pred)),
            "cv_mae_mean": getattr(self, "cv_mae_mean_", None),
            "cv_mae_std": getattr(self, "cv_mae_std_", None),
        }
        return self.metrics_

    def feature_importance(self) -> dict:
        """Return importance/coefficient if model support it, else {}."""
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(REGRESSION_FEATURES, self.model.feature_importances_.tolist()))
        if hasattr(self.model, "coef_"):
            return dict(zip(REGRESSION_FEATURES, self.model.coef_.tolist()))
        return {}


class LinearRegressionModel(BaseRegressor):
    """Simple linear baseline - every other model must beat this."""

    def __init__(self):
        super().__init__("queue_linear_regression")
        self.model = LinearRegression()


class RandomForestRegressorModel(BaseRegressor):
    """Random Forest regressor - capture non-linear interaction."""

    def __init__(self):
        super().__init__("queue_random_forest_regressor")
        self.model = RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=config.ML_RANDOM_STATE,
            n_jobs=-1,
        )


class GradientBoostingRegressorModel(BaseRegressor):
    """Gradient Boosting regressor - usually best accuracy of the three."""

    def __init__(self):
        super().__init__("queue_gradient_boosting_regressor")
        self.model = GradientBoostingRegressor(
            n_estimators=config.GB_N_ESTIMATORS,
            learning_rate=config.GB_LEARNING_RATE,
            random_state=config.ML_RANDOM_STATE,
        )
