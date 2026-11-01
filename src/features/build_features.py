import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.clean_data import PSEUDO_CATEGORICAL_COLUMNS


def get_feature_columns(df: pd.DataFrame, target_column: str):
    """Split features into numeric vs. categorical by dtype, except for the
    coded-category columns (e.g. admission_type_id) which must stay
    categorical even when their dtype is numeric — which happens whenever
    this runs on a CSV round-tripped through disk rather than the live
    DataFrame `clean()` just cast to string."""
    feature_df = df.drop(columns=[target_column])
    numeric_cols = feature_df.select_dtypes(include="number").columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude="number").columns.tolist()

    for col in PSEUDO_CATEGORICAL_COLUMNS:
        if col in numeric_cols:
            numeric_cols.remove(col)
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])
