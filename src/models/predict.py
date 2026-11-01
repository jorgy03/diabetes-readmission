import joblib
import pandas as pd

from src.config import MODELS_DIR


def load_model(model_name: str = "random_forest.joblib"):
    return joblib.load(MODELS_DIR / model_name)


def predict(model, X: pd.DataFrame):
    return model.predict(X)


def predict_proba(model, X: pd.DataFrame):
    return model.predict_proba(X)[:, 1]
