# Data

This folder is not version controlled (see `.gitignore`) — the dataset is too large and is publicly redistributable from the source below.

## Getting the data

1. Download the **Diabetes 130-US Hospitals for years 1999-2008** dataset from the UCI Machine Learning Repository:
   https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
2. Extract `diabetic_data.csv` into `data/raw/`.

## Layout

- `raw/` — the untouched CSV as downloaded.
- `processed/` — cleaned/feature-engineered CSVs produced by `notebooks/02_preprocessing.ipynb` (via `src/data/clean_data.py`).
- `external/` — any supplementary data (e.g. ICD-9 code mappings), not part of the original dataset.
