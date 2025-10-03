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
   ├─ raw/          # put datasets here locally (not committed)
   └─ processed/    # derived files/splits (not committed)
```

## Setup

python -m venv .venv

# Windows: .venv\Scripts\activate

# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

## Data

Datasets are not committed. Place them locally:

CSV with columns text and label → data/raw/your_dataset.csv

Or plain text files under folders → data/raw/class_a/*.txt, data/raw/class_b/*.txt

Update the notebooks’ paths accordingly.

## Tips
Keep notebooks output-free before committing (Colab: Edit ▸ Clear all outputs).

If you later need more libs (e.g., spaCy/transformers), add them to requirements.txt.
