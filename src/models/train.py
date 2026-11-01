import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.config import MODELS_DIR, RANDOM_STATE, TARGET_COLUMN
from src.features.build_features import build_preprocessor, get_feature_columns


def split_data(df: pd.DataFrame, target_column: str = TARGET_COLUMN):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Random Forest pipeline. min_samples_leaf caps tree depth — with the
    default of 1, trees grow until every leaf is pure/single-sample and
    memorize the training set, which swamps whatever class_weight="balanced"
    is trying to do to the decision boundary."""
    numeric_cols, categorical_cols = get_feature_columns(X.assign(**{TARGET_COLUMN: 0}), TARGET_COLUMN)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_gb_pipeline(X: pd.DataFrame) -> Pipeline:
    """Gradient boosting pipeline. GradientBoostingClassifier has no
    class_weight param, so imbalance is handled via sample_weight at fit time
    (see train_and_save)."""
    numeric_cols, categorical_cols = get_feature_columns(X.assign(**{TARGET_COLUMN: 0}), TARGET_COLUMN)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_and_save(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, model_name: str):
    """Fit a pipeline (passing balanced sample weights for models without a
    class_weight param, e.g. GradientBoostingClassifier) and save it."""
    fit_kwargs = {}
    if not hasattr(pipeline.named_steps["model"], "class_weight"):
        fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_train)

    pipeline.fit(X_train, y_train, **fit_kwargs)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / model_name)

    return pipeline
