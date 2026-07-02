"""
model_orchestrator.py
----------------------
Facade that trains every model required by rubric item C in one call,
collects metric into a single tidy report, save trained model file and
write outputs/model_metrics.csv (one row per model so it easy to put
straight into a report table).
"""

from __future__ import annotations

import json

import pandas as pd

from src.models.classification_models import (
    GradientBoostingCongestionClassifier, RandomForestCongestionClassifier,
)
from src.models.clustering import TrafficStateClusterer
from src.models.incident_detector import (
    LogisticRegressionIncidentDetector, RandomForestIncidentDetector,
)
from src.models.regression_models import (
    GradientBoostingRegressorModel, LinearRegressionModel, RandomForestRegressorModel,
)
from src.utils import config
from src.utils.logger import log


class ModelTrainingOrchestrator:
    """Train, evaluate and persist EVERY model used in this project."""

    def __init__(self):
        self.regressors = {
            "Linear_Regression": LinearRegressionModel(),
            "Random_Forest_Regressor": RandomForestRegressorModel(),
            "Gradient_Boosting_Regressor": GradientBoostingRegressorModel(),
        }
        self.classifiers = {
            "Random_Forest_Classifier": RandomForestCongestionClassifier(),
            "Gradient_Boosting_Classifier": GradientBoostingCongestionClassifier(),
        }
        self.incident_detectors = {
            "Logistic_Regression_Incident": LogisticRegressionIncidentDetector(),
            "Random_Forest_Incident": RandomForestIncidentDetector(),
        }
        self.clusterer = TrafficStateClusterer()
        self.report: dict = {}

    def run(self, df: pd.DataFrame) -> dict:
        log("Train", "Training regression model (queue length prediction)...")
        for name, model in self.regressors.items():
            model.train(df)
            model.save()
            log("Train", f"  {name}: MAE={model.metrics_['mae']:.2f}, "
                          f"RMSE={model.metrics_['rmse']:.2f}, R2={model.metrics_['r2']:.3f}")

        log("Train", "Training congestion classification model...")
        for name, model in self.classifiers.items():
            model.train(df)
            model.save()
            log("Train", f"  {name}: accuracy={model.metrics_['accuracy']:.3f}, "
                          f"F1={model.metrics_['f1_weighted']:.3f}")

        log("Train", "Training incident detection model (recall-tuned)...")
        for name, model in self.incident_detectors.items():
            model.train(df)
            model.save()
            log("Train", f"  {name}: recall={model.metrics_['tuned_recall']:.3f}, "
                          f"precision={model.metrics_['tuned_precision']:.3f}")

        log("Train", "Running K-Means traffic state clustering...")
        cluster_labels = self.clusterer.fit_predict(df)
        self.clusterer.save()
        log("Train", f"  silhouette score={self.clusterer.metrics_['silhouette_score']:.3f}")

        self.report = {
            "regression": {name: m.metrics_ for name, m in self.regressors.items()},
            "classification": {name: m.metrics_ for name, m in self.classifiers.items()},
            "incident_detection": {name: m.metrics_ for name, m in self.incident_detectors.items()},
            "clustering": self.clusterer.metrics_,
            "incident_tradeoff_explanation": list(self.incident_detectors.values())[0].explain_tradeoff(),
        }

        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.OUTPUT_DIR / "metrics.json", "w") as f:
            json.dump(self.report, f, indent=2)

        self._write_metrics_csv()
        return self.report

    def best_classifier(self):
        """Return classifier object with highest weighted F1, used by
        api.py / dashboard for live prediction."""
        return max(self.classifiers.values(), key=lambda m: m.metrics_["f1_weighted"])

    def best_regressor(self):
        return min(self.regressors.values(), key=lambda m: m.metrics_["mae"])

    def best_incident_detector(self):
        return max(self.incident_detectors.values(), key=lambda m: m.metrics_["tuned_recall"])

    def _write_metrics_csv(self) -> None:
        rows = []
        for name, m in self.report["regression"].items():
            rows.append({"model_type": "Regression", "model_name": name,
                         "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"],
                         "accuracy": None, "precision": None, "recall": None, "f1": None})
        for name, m in self.report["classification"].items():
            rows.append({"model_type": "Classification", "model_name": name,
                         "mae": None, "rmse": None, "r2": None,
                         "accuracy": m["accuracy"], "precision": m["precision_weighted"],
                         "recall": m["recall_weighted"], "f1": m["f1_weighted"]})
        for name, m in self.report["incident_detection"].items():
            rows.append({"model_type": "Incident_Detection", "model_name": name,
                         "mae": None, "rmse": None, "r2": None,
                         "accuracy": None, "precision": m["tuned_precision"],
                         "recall": m["tuned_recall"], "f1": m["tuned_f1"]})
        pd.DataFrame(rows).to_csv(config.OUTPUT_DIR / "model_metrics.csv", index=False)
        log("Train", f"Saved outputs/model_metrics.csv ({len(rows)} model rows)")
