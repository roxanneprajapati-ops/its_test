"""
conftest.py
-----------
Make sure project root is on sys.path so `from src...` import work no
matter where pytest is invoked from (no fragile path hack needed in
every test file).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
