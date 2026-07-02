"""
logger.py
---------
Small helper so every module print message in same style. Not a big
logging framework, just simple consistent function, because project
need to stay easy to demo and explain in 15 minute.
"""

from __future__ import annotations

import datetime as dt


def log(stage: str, message: str) -> None:
    """Print one timestamped line. Use this instead of plain print()
    so terminal output look professional during live demo."""
    now = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{stage}] {message}")
