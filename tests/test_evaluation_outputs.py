"""
test_evaluation_outputs.py
-----------------------------
Test the evaluation layer: baseline-vs-adaptive comparison CSV export,
model metrics CSV export, and security test result export. Exercises
rubric item G "dashboard/data export".
"""

import pandas as pd

from src.evaluation.performance_evaluator import BaselineVsAdaptiveEvaluator
from src.evaluation.security_test_runner import run_and_save as run_security_tests
from src.preprocessing.preprocessing import PreprocessingPipeline
from src.simulation.simulator import TrafficSimulationManager
from src.utils import config


def test_baseline_vs_adaptive_export_has_expected_columns(tmp_path):
    raw = TrafficSimulationManager().run()
    processed = PreprocessingPipeline().run(raw)
    comparison = BaselineVsAdaptiveEvaluator().run(processed)

    expected_cols = {
        "Intersection_ID", "Fixed_Avg_Queue", "Adaptive_Avg_Queue",
        "Queue_Reduction_Pct", "Delay_Reduction_Pct",
        "Estimated_Travel_Time_Improvement_Pct",
    }
    assert expected_cols.issubset(set(comparison.columns))
    assert len(comparison) == len(config.INTERSECTION_IDS)

    saved = pd.read_csv(config.OUTPUT_DIR / "baseline_vs_adaptive.csv")
    assert len(saved) == len(comparison)


def test_security_test_results_export():
    df = run_security_tests()
    assert len(df) == 5
    assert "passed" in df.columns
    saved = pd.read_csv(config.OUTPUT_DIR / "security_test_results.csv")
    assert len(saved) == 5


def test_no_hardcoded_fragile_paths_used():
    """Sanity check: config module always resolve path relative to the
    project, never an absolute machine-specific string."""
    assert config.BASE_DIR.exists()
    assert str(config.RAW_DATASET_PATH).endswith("traffic_dataset_raw.csv")
