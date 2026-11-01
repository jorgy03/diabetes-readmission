import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def get_feature_importances(pipeline: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """Return the top_n most important features for a fitted pipeline.

    Works for tree-based models (.feature_importances_) and linear models
    (.coef_, ranked by absolute value).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError(f"Model {type(model)} has no feature_importances_ or coef_")

    clean_names = [name.split("__", 1)[-1] for name in feature_names]

    return (
        pd.DataFrame({"feature": clean_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
