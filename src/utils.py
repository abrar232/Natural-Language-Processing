from __future__ import annotations
from pathlib import Path
import random, numpy as np

def project_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "data": root / "data",
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "notebooks": root / "notebooks",
        "src": root / "src",
    }

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
