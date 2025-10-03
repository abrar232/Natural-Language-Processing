from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import pandas as pd

def load_csv_text_label(csv_path: str | Path, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Load a CSV with 'text' and 'label' columns."""
    df = pd.read_csv(csv_path)
    if text_col not in df or label_col not in df:
        raise ValueError(f"CSV must contain '{text_col}' and '{label_col}' columns.")
    return df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

def load_txt_folders(root: str | Path) -> pd.DataFrame:
    """
    Load plain .txt files from class folders:
    root/class_a/*.txt, root/class_b/*.txt, ...
    Returns DataFrame with columns: text, label, path
    """
    root = Path(root)
    rows = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = class_dir.name
        for fp in class_dir.rglob("*.txt"):
            rows.append({"text": fp.read_text(encoding="utf-8", errors="ignore"), "label": label, "path": str(fp)})
    return pd.DataFrame(rows)

def train_val_split(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic split on row order (after shuffle)."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(len(df) * val_ratio))
    val_df = df.iloc[:n_val].reset_index(drop=True)
    train_df = df.iloc[n_val:].reset_index(drop=True)
    return train_df, val_df

def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
