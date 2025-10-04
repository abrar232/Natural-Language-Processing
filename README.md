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
    
    # If not listed in your requirements, install these:
    pip install datasets "pyarrow>=14" pandas numpy scikit-learn tqdm
    
    # (Optional for sequence-label metrics)
    pip install seqeval

## Data

Primary source: Hugging Face dataset surrey-nlp/PLOD-CW (coursework subset for token classification).

### Load at runtime (recommended)

    from datasets import load_dataset
    ds = load_dataset("surrey-nlp/PLOD-CW", cache_dir="data/raw/hf_cache")
    print(ds)  # DatasetDict with train/validation/test

### Label names (int → string)

    label_names = ds["train"].features["ner_tags"].feature.names
    print(label_names)  # e.g. ["B-O", "B-AC", "B-LF", "I-LF", ...]

### Quickstart (put at the top of each notebook)

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
    print(sample["ner_tags"][:20])  # ints → decode with label_names(ds)

### Decode tags to strings (handy for inspection/metrics)

    names = label_names(ds)
    def decode_tags(int_tags):
        return [names[i] for i in int_tags]
        
    print(decode_tags(ds["train"][0]["ner_tags"][:20]))

## Minimal src/ API

- src/utils.py

  - project_paths() → common paths relative to repo root

  - set_seed(seed) → reproducible RNG (Python + NumPy)

- src/dataio.py

  - load_plod_cw(cache_dir="data/raw/hf_cache") → datasets.DatasetDict
  
  - label_names(ds) → list of tag strings
  
  - export_parquet(ds, out_dir="data/raw") → write split Parquet files
  
  - export_conll(ds, out_dir="data/raw") → write CoNLL files
  
  - read_conll(path) → pandas.DataFrame with tokens / tags

## Tips & Gotchas

  - Parquet errors? pip install "pyarrow>=14"
  
  - Clean diffs: keep notebooks output-free before committing (Colab: Edit → Clear all outputs).
  
  - Rename for clarity: feel free to rename Server (2).ipynb to something clearer.
