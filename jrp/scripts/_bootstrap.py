"""Put the repository root on sys.path so `harness` and `baselines` import.

The repository is driven by scripts rather than installed, so every entry point
imports this first.
"""

import sys
from pathlib import Path

root = str(Path(__file__).resolve().parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)
