# NLP Coursework — PLOD-CW (Abbreviation Detection)

Two notebooks exploring token-level abbreviation detection using the **PLOD-CW** dataset (Hugging Face).  
The repo is intentionally lightweight: just notebooks, a tiny `src/` helper module, and empty `data/` placeholders.  
Datasets are **loaded at runtime** via `datasets.load_dataset(...)` and cached locally (not committed).

---

## Project Structure
```text
.
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ notebooks/
│  ├─ Server (2).ipynb      # utility/experiments (rename if you prefer)
│  └─ nlp_cw1v1.ipynb       # main coursework notebook
├─ src/
│  ├─ __init__.py
│  ├─ utils.py              # paths + seeding (minimal)
│  └─ dataio.py             # PLOD-CW loaders + optional export helpers
└─ data/
No models/, logs/, or outputs/ folders are created—this repo keeps only what’s required.
```

## Setup

python -m venv .venv

# Windows: .venv\Scripts\activate

# macOS/Linux: source .venv/bin/activate

# Core libs

pip install -r requirements.txt

# If you don't have it listed, install datasets + parquet support:

pip install datasets "pyarrow>=14"

# (Optional for sequence-label metrics)

pip install seqeval

## Data

Primary source: Hugging Face surrey-nlp/PLOD-CW (coursework subset for token classification).

You don’t commit dataset files; they’re cached locally under data/raw/hf_cache/.

Load at runtime (recommended)

from datasets import load_dataset

ds = load_dataset("surrey-nlp/PLOD-CW", cache_dir="data/raw/hf_cache")

print(ds)  # e.g., DatasetDict with train/validation/test

Label names (int → string)

label_names = ds["train"].features["ner_tags"].feature.names

print(label_names)  # e.g. ["B-O", "B-AC", "B-LF", "I-LF", ...]

Optional: export local copies (kept out of Git)

Use the helpers in src/dataio.py if you want Parquet/CoNLL files under data/raw/:


from src.dataio import load_plod_cw, export_parquet, export_conll, label_names

ds = load_plod_cw(cache_dir="data/raw/hf_cache")  # identical to load_dataset(...)

# Write Parquet: train-00000-of-00001.parquet, validation-..., test-...

export_parquet(ds, out_dir="data/raw")

# Write CoNLL: data_train.conll, data_dev.conll, data_test.conll

export_conll(ds, out_dir="data/raw")

Note: data/ is gitignored except for .gitkeep, so the cache/exports won’t bloat your repo.

Quickstart (put at the top of each notebook)

from src.utils import project_paths, set_seed

from src.dataio import load_plod_cw, label_names

set_seed(42)

P = project_paths()

print("Data root:", P["raw"])

ds = load_plod_cw(cache_dir=P["raw"] / "hf_cache")

print(ds)

print("Labels:", label_names(ds))

# Peek a sample

sample = ds["train"][0]

print(sample["tokens"][:20])

print(sample["ner_tags"][:20])  # ints → use label_names(...) to decode

Decoding tags to strings (for inspection/metrics)

names = label_names(ds)

def decode_tags(int_tags): return [names[i] for i in int_tags]

print(decode_tags(ds["train"][0]["ner_tags"][:20]))

(Optional) seqeval report (if you later produce predictions)

# pip install seqeval

from seqeval.metrics import classification_report

# gold and preds are lists of tag-strings per sentence, e.g.:

# gold = [["B-AC", "B-O", ...], ...]

# preds = [["B-AC", "B-O", ...], ...]

print(classification_report(gold, preds, digits=4))

Minimal src/ API

src/utils.py

project_paths() → common paths relative to repo root

set_seed(seed) → reproducible RNG (Python + NumPy)

src/dataio.py

load_plod_cw(cache_dir="data/raw/hf_cache") → datasets.DatasetDict

label_names(ds) → list of tag strings

export_parquet(ds, out_dir="data/raw") → write split Parquet files

export_conll(ds, out_dir="data/raw") → write CoNLL files

read_conll(path) → DataFrame with tokens/tags

## Tips & Gotchas

Parquet errors: install pyarrow>=14.

Notebook outputs: keep notebooks output-free before committing (clean diffs).

Renaming notebooks: feel free to rename Server (2).ipynb to something clearer (e.g., utils_experiments.ipynb).
