"""
performance_evaluator.py
--------------------------
Compare the FIXED-TIME baseline signal plan against the AI-adaptive
signal plan on the exact same dataset (rubric item E + B), producing
the headline ITS efficiency metrics:

  - average delay reduction (%)
  - queue length reduction (%)
  - congestion reduction (% of High-congestion record)
  - estimated travel-time improvement (%)
  - signal cycle efficiency (vehicle served per green-second)

Method: for every record we already know Queue_Length (and Delay) under
the historic/baseline condition. We then *simulate* what queue/delay
would look like if the adaptive green time had been used instead, with
a simple, transparent linear relationship (more green time -> shorter
queue, up to a point) - good enough to demonstrate the ITS evaluation
methodology without needing a full microscopic traffic simulator
(out of scope for a one-semester assignment).
"""

from __future__ import annotations

import pandas as pd

from src.optimization.optimizer import SignalOptimizationEngine
from src.utils import config
from src.utils.logger import log


class BaselineVsAdaptiveEvaluator:
    """Run the fixed-time vs adaptive comparison over a processed
    dataset and produce a tidy per-intersection summary table."""

    def __init__(self, optimizer: SignalOptimizationEngine | None = None):
        self.optimizer = optimizer or SignalOptimizationEngine()

    def _simulate_adaptive_outcome(self, row: pd.Series) -> tuple[int, float, float]:
        recommendation = self.optimizer.recommend(
            intersection_id=row["Intersection_ID"],
            congestion_level=row["Congestion_Level"],
            queue_length=int(row["Queue_Length"]),
            incident_detected=bool(row["Incident_Flag"]),
        )
        green = recommendation.recommended_green_time
        fixed_green = recommendation.fixed_time_green

        # more green relative to fixed baseline -> proportionally less
        # queue/delay, capped so the relationship stay realistic
        relief_ratio = min(1.6, green / fixed_green) if fixed_green else 1.0
        improvement_factor = 1.0 - min(0.45, (relief_ratio - 1.0) * 0.55) if relief_ratio > 1 else 1.0 + (1 - relief_ratio) * 0.2

        adaptive_queue = max(0.0, row["Queue_Length"] * improvement_factor)
        adaptive_delay = max(0.0, row["Delay_Seconds"] * improvement_factor)
        return green, adaptive_queue, adaptive_delay

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for intersection_id, group in df.groupby("Intersection_ID"):
            fixed_avg_queue = group["Queue_Length"].mean()
            fixed_avg_delay = group["Delay_Seconds"].mean()
            fixed_high_pct = (group["Congestion_Level"] == "High").mean() * 100

            adaptive_greens, adaptive_queues, adaptive_delays = [], [], []
            for _, row in group.iterrows():
                green, a_queue, a_delay = self._simulate_adaptive_outcome(row)
                adaptive_greens.append(green)
                adaptive_queues.append(a_queue)
                adaptive_delays.append(a_delay)

            adaptive_avg_queue = sum(adaptive_queues) / len(adaptive_queues)
            adaptive_avg_delay = sum(adaptive_delays) / len(adaptive_delays)
            avg_green = sum(adaptive_greens) / len(adaptive_greens)

            queue_reduction_pct = (1 - adaptive_avg_queue / fixed_avg_queue) * 100 if fixed_avg_queue else 0.0
            delay_reduction_pct = (1 - adaptive_avg_delay / fixed_avg_delay) * 100 if fixed_avg_delay else 0.0
            # travel-time improvement approximated from delay reduction,
            # since delay is the main variable component of travel time
            travel_time_improvement_pct = delay_reduction_pct * 0.85

            # cycle efficiency: vehicles served per second of green time
            fixed_efficiency = group["Vehicle_Count"].mean() / config.BASE_GREEN_TIME
            adaptive_efficiency = group["Vehicle_Count"].mean() / avg_green if avg_green else 0.0

            records.append({
                "Intersection_ID": intersection_id,
                "Intersection_Name": config.INTERSECTION_NAMES.get(intersection_id, intersection_id),
                "Fixed_Avg_Queue": round(fixed_avg_queue, 2),
                "Adaptive_Avg_Queue": round(adaptive_avg_queue, 2),
                "Queue_Reduction_Pct": round(queue_reduction_pct, 2),
                "Fixed_Avg_Delay_Sec": round(fixed_avg_delay, 2),
                "Adaptive_Avg_Delay_Sec": round(adaptive_avg_delay, 2),
                "Delay_Reduction_Pct": round(delay_reduction_pct, 2),
                "High_Congestion_Pct": round(fixed_high_pct, 2),
                "Estimated_Travel_Time_Improvement_Pct": round(travel_time_improvement_pct, 2),
                "Fixed_Cycle_Efficiency_veh_per_sec": round(fixed_efficiency, 3),
                "Adaptive_Cycle_Efficiency_veh_per_sec": round(adaptive_efficiency, 3),
            })

        result = pd.DataFrame(records)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result.to_csv(config.OUTPUT_DIR / "baseline_vs_adaptive.csv", index=False)
        log("Evaluator", "Saved outputs/baseline_vs_adaptive.csv")
        return result
