"""
main.py
-------
Single entry point that run the whole ITS pipeline end to end:

    1. Simulate raw traffic sensor data        (sensing layer)
    2. Generate SHA-256 integrity hash          (security layer)
    3. Clean + feature engineer dataset         (preprocessing)
    4. Train & compare every ML model           (analytics layer)
    5. Compare fixed-time vs adaptive signal    (optimization layer)
    6. Run simulated cyberattack test suite     (security layer)
    7. Generate figures for report/dashboard

Run with:
    python main.py
"""

from __future__ import annotations

from src.evaluation.figures import FigureGenerator
from src.evaluation.model_orchestrator import ModelTrainingOrchestrator
from src.evaluation.performance_evaluator import BaselineVsAdaptiveEvaluator
from src.evaluation.security_test_runner import run_and_save as run_security_tests
from src.preprocessing.preprocessing import PreprocessingPipeline
from src.security.security import FileIntegrityChecker
from src.simulation.simulator import TrafficSimulationManager
from src.utils import config
from src.utils.logger import log


def print_target_comparison(report: dict) -> None:
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS vs PROJECT TARGETS")
    print("=" * 70)

    def status(actual, target, higher_is_better=True):
        ok = (actual >= target) if higher_is_better else (actual <= target)
        return "PASS" if ok else "BELOW TARGET"

    best_clf = max(report["classification"].values(), key=lambda m: m["f1_weighted"])
    best_reg = min(report["regression"].values(), key=lambda m: m["mae"])
    best_inc = max(report["incident_detection"].values(), key=lambda m: m["tuned_recall"])

    print(f"Best classifier F1     : {best_clf['f1_weighted']:.3f}  "
          f"(target > {config.TARGET_RF_F1}) [{status(best_clf['f1_weighted'], config.TARGET_RF_F1)}]")
    print(f"Best classifier acc.   : {best_clf['accuracy']:.3f}  "
          f"(target > {config.TARGET_RF_ACCURACY}) [{status(best_clf['accuracy'], config.TARGET_RF_ACCURACY)}]")
    print(f"Best regressor MAE     : {best_reg['mae']:.2f}  "
          f"(target < {config.TARGET_REG_MAE}) [{status(best_reg['mae'], config.TARGET_REG_MAE, False)}]")
    print(f"Best regressor RMSE    : {best_reg['rmse']:.2f}  "
          f"(target < {config.TARGET_REG_RMSE}) [{status(best_reg['rmse'], config.TARGET_REG_RMSE, False)}]")
    print(f"Best incident recall   : {best_inc['tuned_recall']:.3f}  "
          f"(target > {config.TARGET_INCIDENT_RECALL}) [{status(best_inc['tuned_recall'], config.TARGET_INCIDENT_RECALL)}]")
    print("=" * 70)


def main() -> None:
    log("Main", "[1/7] Simulating traffic sensor data for 4 Auckland intersections...")
    sim_manager = TrafficSimulationManager()
    raw_df = sim_manager.save()

    log("Main", "[2/7] Generating SHA-256 dataset integrity hash...")
    digest = FileIntegrityChecker().generate_hash_file()
    log("Main", f"      SHA-256: {digest}")

    log("Main", "[3/7] Cleaning data and engineering feature/label...")
    pipeline = PreprocessingPipeline()
    processed_df = pipeline.run(raw_df)
    pipeline.save(processed_df)
    log("Main", f"      Processed dataset shape: {processed_df.shape}")

    log("Main", "[4/7] Training and comparing every ML model...")
    orchestrator = ModelTrainingOrchestrator()
    report = orchestrator.run(processed_df)

    log("Main", "[5/7] Comparing fixed-time baseline vs AI-adaptive signal...")
    comparison_df = BaselineVsAdaptiveEvaluator().run(processed_df)
    print(comparison_df.to_string(index=False))

    log("Main", "[6/7] Running simulated cyberattack test suite...")
    security_df = run_security_tests()
    print(security_df[["attack_name", "passed", "reason"]].to_string(index=False))

    log("Main", "[7/7] Generating figures for report and dashboard...")
    FigureGenerator().generate_all(processed_df, report, comparison_df)

    log("Main", "Pipeline complete. See outputs/ for metrics, CSV and figures.")
    print_target_comparison(report)


if __name__ == "__main__":
    main()
