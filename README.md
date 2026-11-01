# Predicting 30-Day Hospital Readmission in Diabetes Patients

Predicting whether a diabetic patient will be readmitted to the hospital within 30 days, using the [UCI Diabetes 130-US Hospitals (1999-2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) dataset.

## Result

**Gradient Boosting, ROC-AUC 0.64 vs. a 0.50 "always predict no" baseline — catches 55% of actual 30-day readmissions (recall), up from 0% for the baseline.**

| Model | ROC-AUC | Recall (readmit) | Precision (readmit) |
|---|---|---|---|
| Always "no" (baseline) | 0.500 | 0.000 | 0.000 |
| Logistic Regression | 0.637 | 0.528 | 0.139 |
| Random Forest (`class_weight="balanced"`, `min_samples_leaf=50`) | 0.639 | 0.552 | 0.137 |
| **Gradient Boosting** | **0.640** | **0.547** | 0.137 |

All three real models land in a tight band (ROC-AUC 0.637–0.640) — Gradient Boosting is only marginally ahead of Random Forest and Logistic Regression, not a clear winner. Precision is low across the board (~14%) because the positive class is rare — at this recall level the model is intentionally over-flagging to avoid missing readmissions, which is usually the right trade-off for a clinical screening tool (a missed readmission is costlier than an extra follow-up call).

![Target class balance](reports/figures/01_target_balance.png)
![Feature importance](reports/figures/04_feature_importance.png)

## Problem

Hospital readmissions within 30 days are costly and often considered a signal of poor care continuity. This project frames the original 3-class `readmitted` label (`NO`, `>30`, `<30`) as a **binary classification problem**: did the patient return within 30 days or not.

The raw data is ~11% positive (chart above). After cleaning — keeping one encounter per patient and dropping encounters where the patient died — the modeling population's actual rate is **~8.9%** (6,296 / 70,439 rows). These are different populations and shouldn't be used interchangeably: the raw-encounter rate describes the dataset as collected, the clean-population rate is what the model is actually trained and evaluated against. Either way, the target class is rare, so the project also deals with **class imbalance**.

## Project structure

```
.
├── data/                   # not version controlled — see data/README.md
│   ├── raw/                # original diabetic_data.csv goes here
│   ├── processed/          # cleaned data, written by notebooks/02_preprocessing.ipynb
│   └── external/           # supplementary reference data (e.g. ID mappings)
├── notebooks/
│   ├── 01_eda.ipynb            # missingness, target balance, distributions
│   ├── 02_preprocessing.ipynb  # cleaning + target construction, writes processed/
│   ├── 03_modeling.ipynb       # baseline, logistic regression, RF, gradient boosting
│   └── 04_evaluation.ipynb     # ROC-AUC, confusion matrix, ROC curve, feature importance
├── src/
│   ├── config.py               # paths, random seed, target column name
│   ├── app.py                  # Streamlit demo — type in patient info, get a risk score
│   ├── data/
│   │   ├── load_data.py        # reads raw CSV (handles "?" as NaN)
│   │   └── clean_data.py       # dedup, drop sparse/ID columns, diagnosis grouping, binary target
│   ├── features/
│   │   └── build_features.py   # numeric/categorical preprocessing pipeline
│   ├── models/
│   │   ├── train.py             # train/test split + Random Forest / Gradient Boosting pipelines
│   │   ├── baseline.py          # "always no" dummy baseline + Logistic Regression baseline
│   │   ├── predict.py           # load a saved model and score new data
│   │   ├── evaluate.py          # ROC-AUC, classification report, confusion matrix
│   │   └── explain.py           # feature importance extraction
│   └── visualization/
│       └── visualize.py         # class balance, missingness, confusion matrix, ROC curve, feature importance plots
├── models/                 # saved model artifacts (.joblib), not version controlled
├── reports/figures/        # exported plots
├── tests/                  # unit tests for src/ (pytest)
├── requirements.txt
└── pyproject.toml
```

The intent: **notebooks are for exploration and narrative** (what did I find, what did I decide and why), while **`src/` holds the reusable, tested logic** that notebooks, the Streamlit app, and any future script call into. Nothing important should exist only inside a notebook cell.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # on Windows
pip install -r requirements.txt
pip install -e .                # makes `src` importable as a package
```

Then follow `data/README.md` to download the dataset into `data/raw/`.

## Workflow

Run the notebooks in order: `01_eda` → `02_preprocessing` → `03_modeling` → `04_evaluation`. Each one imports from `src/` rather than redefining logic inline.

Run tests with:

```bash
pytest
```

Run the demo app (after `03_modeling` has produced `models/best_model.joblib`):

```bash
streamlit run src/app.py
```

## Key design decisions

- **Patient-level leakage guard**: the same `patient_nbr` can have many encounters. If two encounters from the same patient land on opposite sides of the train/test split, the model can partly "recognize" that patient rather than generalize, inflating test performance. `drop_duplicate_patients` keeps only the first encounter per patient before splitting — "first" meaning lowest `encounter_id` (the closest thing this dataset has to a timestamp). Sorting by `patient_nbr` alone isn't enough: pandas' default sort isn't guaranteed stable, so ties among a patient's rows can land in any order and silently keep a later visit instead of the first. `encounter_id` as a secondary sort key makes the choice deterministic; see the test in `tests/test_clean_data.py` that specifically exercises scrambled/interleaved input to catch a regression here.
- **Expired-patient filter**: `discharge_disposition_id` codes 11/19/20/21 mean the patient died during the stay — they cannot be "readmitted," so leaving them in is pure label noise. They're dropped in `clean()`.
- **Coded IDs aren't numbers**: `admission_type_id`, `discharge_disposition_id`, and `admission_source_id` are integer-coded categories (e.g. `1` = "Emergency"), not magnitudes. Left as-is they'd get silently standard-scaled as if "8" were twice "4" — they're explicitly treated as categorical throughout the pipeline (`PSEUDO_CATEGORICAL_COLUMNS` in `build_features.py`), including after a CSV round-trip, which re-infers them as `int64` and would otherwise undo the fix.
- **Diagnosis grouping**: `diag_1/2/3` are raw ICD-9 codes — hundreds of distinct values, too high-cardinality to one-hot usefully. They're bucketed into clinical categories (circulatory, respiratory, digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasms, other), following the standard approach from the dataset's original paper (Strack et al., 2014).
- **Class imbalance**: `class_weight="balanced"` for Random Forest / Logistic Regression; `GradientBoostingClassifier` has no `class_weight` param, so it gets balanced `sample_weight` at fit time instead. Class weighting alone isn't sufficient, though — the first Random Forest attempt used the default `min_samples_leaf=1`, which grows every tree down to pure single-sample leaves. That overfits so hard it memorizes the training set regardless of how the loss is weighted, and recall was 0.01 as a result. Capping `min_samples_leaf=50` forces shallower, more general trees and brought recall to 0.55 (see the results table above) — a reminder that imbalance handling and model capacity interact, and a "balanced" flag isn't a substitute for checking the model isn't just overfitting.
- **The model's output is a score, not a calibrated probability**: training with `class_weight`/`sample_weight` rebalancing systematically shifts predicted probabilities upward relative to the true base rate, specifically to make the decision threshold catch more positives. That means `predict_proba` numbers can't be read as literal readmission odds, and shouldn't be compared directly to the dataset's true base rate (~8.9%) — doing so overstates risk. The Streamlit app labels its output a **risk score** rather than a probability for this reason; calibrating it properly (e.g. `CalibratedClassifierCV`) is a natural next step if a true probability is needed.
- **ROC-AUC alone is a weak signal under imbalance**: recall and precision are tracked alongside it, since a model can have a decent ROC-AUC while still being useless at the operating threshold that matters.

## Next steps / ideas

- Calibrate predicted probabilities (`CalibratedClassifierCV`) so the model's output can be read as an actual probability instead of a relative risk score.
- Try SMOTE (via `imbalanced-learn`, already a dependency) and compare against class-weighting.
- Threshold tuning: pick a decision threshold off a precision-recall curve instead of the default 0.5, based on the real cost of a missed readmission vs. an unnecessary follow-up.
- SHAP values for per-patient explanations (the current feature importance is global, not per-prediction).
- Hyperparameter tuning via `GridSearchCV` or `Optuna`.
- Deploy `src/app.py` on [Streamlit Community Cloud](https://streamlit.io/cloud) (free) for a live demo link.

## Data source & citation

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository.
