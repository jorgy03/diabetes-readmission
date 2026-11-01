import pandas as pd

from src.config import TARGET_COLUMN

# Columns that are near-constant or near-entirely missing in this dataset.
COLUMNS_TO_DROP = ["weight", "payer_code", "medical_specialty"]

# Pure identifiers: useful for dedup (patient_nbr) or nothing at all
# (encounter_id), but meaningless as model features.
ID_COLUMNS = ["encounter_id", "patient_nbr"]

# These are coded categories (e.g. 1 = "Emergency"), not magnitudes — stored
# as ints in the raw CSV, so they must be cast to string or a tree/linear
# model will treat them as ordered/scaled numeric features.
PSEUDO_CATEGORICAL_COLUMNS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

# discharge_disposition_id codes meaning the patient died during the
# encounter (see data/raw/IDS_mapping.csv). They cannot be "readmitted", so
# keeping them in only adds label noise.
EXPIRED_DISCHARGE_CODES = [11, 19, 20, 21]

# ICD-9 code -> clinical category, following the bucketing used in Strack et
# al. (2014), the paper this dataset comes from.
_DIAG_RANGES = [
    (390, 459, "circulatory"),
    (785, 785, "circulatory"),
    (460, 519, "respiratory"),
    (786, 786, "respiratory"),
    (520, 579, "digestive"),
    (787, 787, "digestive"),
    (580, 629, "genitourinary"),
    (788, 788, "genitourinary"),
    (800, 999, "injury"),
    (710, 739, "musculoskeletal"),
    (140, 239, "neoplasms"),
    (240, 279, "other"),  # endocrine/metabolic other than diabetes itself
    (680, 709, "other"),
    (782, 782, "other"),
    (1, 139, "other"),
    (290, 319, "other"),
    (280, 289, "other"),
    (320, 389, "other"),
    (630, 679, "other"),
    (740, 759, "other"),
    (760, 779, "other"),
    (780, 781, "other"),
    (783, 784, "other"),
    (789, 799, "other"),
]


def categorize_diagnosis(code) -> str:
    """Map a single ICD-9 diagnosis code to a clinical category bucket.

    Codes starting with "250" are diabetes. "V" and "E" codes (supplemental
    classification, external causes) and missing codes fall into "other".
    """
    if pd.isna(code):
        return "other"
    code = str(code)
    if code.startswith("250"):
        return "diabetes"
    if code.startswith("V") or code.startswith("E"):
        return "other"
    try:
        numeric_code = float(code)
    except ValueError:
        return "other"
    for low, high, category in _DIAG_RANGES:
        if low <= numeric_code <= high:
            return category
    return "other"


def add_diagnosis_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the high-cardinality diag_1/2/3 ICD-9 codes with category buckets."""
    df = df.copy()
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_cat"] = df[col].apply(categorize_diagnosis)
    return df.drop(columns=["diag_1", "diag_2", "diag_3"])


def cast_pseudo_categorical_columns(df: pd.DataFrame, columns=PSEUDO_CATEGORICAL_COLUMNS) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("Int64").astype(str)
    return df


def drop_expired_patients(df: pd.DataFrame, codes=EXPIRED_DISCHARGE_CODES) -> pd.DataFrame:
    """Drop encounters where the patient died — they can't be readmitted."""
    return df[~df["discharge_disposition_id"].astype("Int64").isin(codes)]


def add_readmission_target(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 3-level `readmitted` column into a binary 30-day target."""
    df = df.copy()
    df[TARGET_COLUMN] = (df["readmitted"] == "<30").astype(int)
    return df.drop(columns=["readmitted"])


def drop_duplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first encounter per patient to avoid leakage across visits.

    Sorting by patient_nbr alone doesn't define which row within a patient's
    group is "first" — pandas' default sort isn't guaranteed stable, so ties
    on patient_nbr can land in any order. encounter_id (a monotonically
    assigned row id, the closest thing this dataset has to a timestamp)
    breaks the tie deterministically.
    """
    return df.sort_values(["patient_nbr", "encounter_id"]).drop_duplicates(subset="patient_nbr", keep="first")


def drop_sparse_columns(df: pd.DataFrame, columns=COLUMNS_TO_DROP) -> pd.DataFrame:
    return df.drop(columns=[c for c in columns if c in df.columns])


def drop_id_columns(df: pd.DataFrame, columns=ID_COLUMNS) -> pd.DataFrame:
    return df.drop(columns=[c for c in columns if c in df.columns])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_sparse_columns(df)
    df = drop_expired_patients(df)
    df = drop_duplicate_patients(df)
    df = add_diagnosis_categories(df)
    df = cast_pseudo_categorical_columns(df)
    df = add_readmission_target(df)
    df = drop_id_columns(df)
    return df.reset_index(drop=True)
