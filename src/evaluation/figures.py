"""
figures.py
----------
Generate the plot file rubric expect in outputs/figures/. Use
matplotlib only (no seaborn dependency) to keep requirements.txt small
and the project light enough to run on a normal laptop.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, just save PNG file
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import config
from src.utils.logger import log


class FigureGenerator:
    """Each method create and save ONE figure. Keep separate so test
    or demo can call just the figure they need."""

    def __init__(self, output_dir=config.FIGURES_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name: str) -> None:
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        log("Figures", f"Saved {path.name}")

    def congestion_distribution(self, df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        df["Congestion_Level"].value_counts().reindex(config.CONGESTION_LABELS).plot(
            kind="bar", ax=ax, color=["#7fbf7f", "#f2c14e", "#e36464"])
        ax.set_title("Congestion Level Distribution")
        ax.set_xlabel("Congestion Level")
        ax.set_ylabel("Record Count")
        self._save(fig, "congestion_distribution")

    def traffic_trend(self, df: pd.DataFrame, intersection_id: str = "INT_01") -> None:
        subset = df[df["Intersection_ID"] == intersection_id].sort_values("Timestamp").head(96 * 3)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(subset["Timestamp"], subset["Vehicle_Count"], label="Vehicle Count")
        ax.plot(subset["Timestamp"], subset["Queue_Length"], label="Queue Length")
        ax.set_title(f"Traffic Trend - {config.INTERSECTION_NAMES.get(intersection_id, intersection_id)}")
        ax.legend()
        fig.autofmt_xdate()
        self._save(fig, f"traffic_trend_{intersection_id}")

    def model_comparison_bar(self, metrics_csv_path=None) -> None:
        path = metrics_csv_path or (config.OUTPUT_DIR / "model_metrics.csv")
        if not path.exists():
            return
        df = pd.read_csv(path)
        clf = df[df["model_type"] == "Classification"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(clf["model_name"], clf["f1"], color="#5b8def")
        ax.set_title("Classifier Comparison (Weighted F1)")
        ax.set_ylabel("F1 Score")
        ax.tick_params(axis="x", rotation=20)
        self._save(fig, "classifier_comparison")

    def incident_precision_recall(self, metrics: dict) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, m in metrics.get("incident_detection", {}).items():
            ax.plot(m["pr_curve_recall"], m["pr_curve_precision"], label=name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Incident Detection: Precision-Recall Curve")
        ax.legend()
        self._save(fig, "incident_precision_recall_curve")

    def baseline_vs_adaptive_chart(self, comparison_df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = comparison_df["Intersection_ID"]
        ax.bar(x, comparison_df["Fixed_Avg_Queue"], width=0.35, label="Fixed-Time", align="edge")
        ax.bar(x, comparison_df["Adaptive_Avg_Queue"], width=-0.35, label="AI-Adaptive", align="edge")
        ax.set_ylabel("Average Queue Length")
        ax.set_title("Fixed-Time vs AI-Adaptive: Average Queue Length")
        ax.legend()
        self._save(fig, "baseline_vs_adaptive_queue")

    def generate_all(self, df: pd.DataFrame, metrics: dict, comparison_df: pd.DataFrame) -> None:
        self.congestion_distribution(df)
        for iid in config.INTERSECTION_IDS:
            self.traffic_trend(df, iid)
        self.model_comparison_bar()
        self.incident_precision_recall(metrics)
        self.baseline_vs_adaptive_chart(comparison_df)
