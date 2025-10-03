## Project Structure
```text
.
├─ notebooks/
│  ├─ Server (2).ipynb
│  └─ nlp_cw1v1.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ utils.py      # paths, seeding
│  └─ dataio.py     # small loaders for CSV/TXT corpora + splits
└─ data/
```

## Setup

python -m venv .venv

# Windows: .venv\Scripts\activate

# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

## Data

This repo uses two formats of the same splits:

- **CoNLL** files: `data_train.conll`, `data_dev.conll`, `data_test.conll`
- **Parquet** files: `train-00000-of-00001.parquet`, `validation-00000-of-00001.parquet`, `test-00000-of-00001.parquet`

## Tips
Keep notebooks output-free before committing (Colab: Edit ▸ Clear all outputs).

If you later need more libs (e.g., spaCy/transformers), add them to requirements.txt.
