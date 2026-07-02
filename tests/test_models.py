"""
test_models.py
----------------
Test model training, prediction output shape and feature importance
for regression, classification and incident detection model.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.classification_models import RandomForestCongestionClassifier
from src.models.clustering import TrafficStateClusterer
from src.models.incident_detector import RandomForestIncidentDetector
from src.models.regression_models import LinearRegressionModel, RandomForestRegressorModel
from src.preprocessing.preprocessing import PreprocessingPipeline
from src.simulation.simulator import TrafficSimulationManager


@pytest.fixture(scope="module")
def processed_df():
    raw = TrafficSimulationManager().run()
    return PreprocessingPipeline().run(raw)


def test_linear_regression_trains_and_predicts(processed_df):
    model = LinearRegressionModel()
    model.train(processed_df)
    preds = model.predict(processed_df.head(10))
    assert len(preds) == 10
    assert "mae" in model.metrics_ and model.metrics_["mae"] >= 0


def test_random_forest_regressor_beats_or_matches_baseline_reasonably(processed_df):
    rf = RandomForestRegressorModel()
    rf.train(processed_df)
    assert rf.metrics_["r2"] > 0.0


def test_congestion_classifier_predicts_known_labels(processed_df):
    clf = RandomForestCongestionClassifier()
    clf.train(processed_df)
    preds = clf.predict(processed_df.head(5))
    assert len(preds) == 5
    assert set(preds).issubset({"Low", "Medium", "High"})
    fi = clf.feature_importance()
    assert len(fi) > 0


def test_incident_detector_outputs_binary_predictions(processed_df):
    det = RandomForestIncidentDetector()
    det.train(processed_df)
    preds = det.predict(processed_df.head(20))
    assert set(np.unique(preds)).issubset({0, 1})
    assert "confusion_matrix" in det.metrics_
    assert det.metrics_["tuned_recall"] >= det.metrics_["default_threshold_recall"] - 1e-6


def test_clustering_returns_one_label_per_row(processed_df):
    clusterer = TrafficStateClusterer(n_clusters=3)
    labels = clusterer.fit_predict(processed_df)
    assert len(labels) == len(processed_df)
    assert labels.nunique() <= 3
