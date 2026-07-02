"""
security_test_runner.py
-------------------------
Run the five simulated attack from src/security/simulated_attacks.py
and write outputs/security_test_results.csv, the evidence file rubric
item J ask for.
"""

from __future__ import annotations

import pandas as pd

from src.security.simulated_attacks import run_all_attacks
from src.utils import config
from src.utils.logger import log


def run_and_save() -> pd.DataFrame:
    results = run_all_attacks()
    df = pd.DataFrame(results)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.OUTPUT_DIR / "security_test_results.csv", index=False)
    passed = int(df["passed"].sum())
    log("Security", f"Saved outputs/security_test_results.csv ({passed}/{len(df)} attacks correctly blocked)")
    return df
