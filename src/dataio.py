"""
Data I/O helpers for the PLOD-CW NLP repo.

- Load the coursework dataset from Hugging Face with a repo-local cache.
- Inspect label names (int → string).
- Optionally export splits to Parquet and/or CoNLL under data/raw/.
- (Bonus) Read a CoNLL file back into a pandas DataFrame.

This keeps the repo lightweight: no large files are committed.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
from src.utils import ensure_dirs
import os

# Optional deps for export/reading; import lazily when used
def _lazy_pd():
    import pandas as pd
    return pd

def _lazy_datasets():
    from datasets import load_dataset, DatasetDict, Dataset
    return load_dataset, DatasetDict, Dataset


# ----------------------------- Core loaders -----------------------------------
def load_plod_cw(cache_dir: str | Path = "data/raw/hf_cache"):
    """
    Load the PLOD-CW dataset (train/validation/test) with a local cache directory.

    Returns
    -------
    datasets.DatasetDict
        Keys typically include 'train', 'validation', 'test'.
    """
    load_dataset, DatasetDict, _ = _lazy_datasets()
    cache_dir = str(cache_dir)
    ensure_dirs(Path(cache_dir))
    ds = load_dataset("surrey-nlp/PLOD-CW", cache_dir=cache_dir)
    return ds


def label_names(ds) -> List[str]:
    """
    Return the string labels for the NER tags (int → tag).
    """
    return ds["train"].features["ner_tags"].feature.names


# ----------------------------- Export helpers ---------------------------------
def export_parquet(ds, out_dir: str | Path = "data/raw") -> None:
    """
    Save each available split to Parquet in `out_dir`.
    """
    pd = _lazy_pd()
    out_dir = Path(out_dir); ensure_dirs(out_dir)
    for split in ("train", "validation", "test"):
        if split in ds:
            df = ds[split].to_pandas()
            (out_dir / f"{split}-00000-of-00001.parquet").parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_dir / f"{split}-00000-of-00001.parquet", index=False)


def _write_conll_split(dset, out_path: Path, tag_names: List[str]) -> None:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in dset:
            tokens = ex["tokens"]
            tags = [tag_names[i] for i in ex["ner_tags"]]
            for tok, tag in zip(tokens, tags):
                f.write(f"{tok} {tag}\n")
            f.write("\n")


def export_conll(ds, out_dir: str | Path = "data/raw") -> None:
    """
    Save each available split to simple CoNLL files in `out_dir`.
    Filenames: data_train.conll, data_dev.conll, data_test.conll
    """
    names = label_names(ds)
    out_dir = Path(out_dir); ensure_dirs(out_dir)
    if "train" in ds:
        _write_conll_split(ds["train"], out_dir / "data_train.conll", names)
    if "validation" in ds:
        _write_conll_split(ds["validation"], out_dir / "data_dev.conll", names)
    if "test" in ds:
        _write_conll_split(ds["test"], out_dir / "data_test.conll", names)


# ----------------------------- Read-back helper -------------------------------
def read_conll(path: str | Path):
    """
    Read a simple 'token TAG' CoNLL file into a pandas DataFrame with
    columns: tokens (List[str]), tags (List[str]).
    """
    pd = _lazy_pd()
    path = Path(path)
    sents, tags = [], []
    cur_toks, cur_tags = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:  # sentence boundary
            if cur_toks:
                sents.append(cur_toks); tags.append(cur_tags)
                cur_toks, cur_tags = [], []
            continue
        parts = line.split()
        cur_toks.append(parts[0])
        cur_tags.append(parts[-1])
    if cur_toks:
        sents.append(cur_toks); tags.append(cur_tags)
    return pd.DataFrame({"tokens": sents, "tags": tags})
