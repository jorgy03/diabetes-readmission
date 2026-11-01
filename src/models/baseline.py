import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import TARGET_COLUMN
from src.features.build_features import build_preprocessor, get_feature_columns


def evaluate_always_no(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Score the trivial "always predict no readmission" baseline."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(pd.DataFrame(index=y_train.index), y_train)
    y_pred = dummy.predict(pd.DataFrame(index=y_test.index))
    y_proba = dummy.predict_proba(pd.DataFrame(index=y_test.index))[:, 1]
    return {"model": dummy, "y_pred": y_pred, "y_proba": y_proba}


def build_logistic_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols, categorical_cols = get_feature_columns(X.assign(**{TARGET_COLUMN: 0}), TARGET_COLUMN)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])
