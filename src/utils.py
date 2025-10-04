"""
Minimal project utilities for the NLP repo.

This repo is intentionally lightweight: just notebooks + tiny helpers.
We expose:
- `project_paths()` → common directories relative to the repo root
- `set_seed()`      → reproducible RNG for Python + NumPy

Assumes this file lives at: <repo>/src/utils.py
"""

from __future__ import annotations
from pathlib import Path
import random, numpy as np


def project_paths() -> dict[str, Path]:
    """
    Return a dict of commonly used paths relative to the **repo root**.

    Keys
    ----
    - root       : Path to the repository root
    - data       : <root>/data
    - raw        : <root>/data/raw
    - processed  : <root>/data/processed
    - notebooks  : <root>/notebooks
    - src        : <root>/src
    """
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
    """
    Seed Python and NumPy RNGs for reproducibility.

    Parameters
    ----------
    seed : int, default=42
        Seed value used for `random` and `numpy`.
    """
    random.seed(seed)
    np.random.seed(seed)
