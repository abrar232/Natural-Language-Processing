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

Data can be accessed using the command:

from datasets import load_dataset

ds = load_dataset("surrey-nlp/PLOD-CW")


## Tips
Keep notebooks output-free before committing (Colab: Edit ▸ Clear all outputs).

If you later need more libs (e.g., spaCy/transformers), add them to requirements.txt.
