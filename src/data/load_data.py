import pandas as pd

from src.config import RAW_DATA_FILE


def load_raw_data(path=RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw UCI Diabetes 130-US Hospitals CSV.

    The dataset encodes missing values as "?" rather than NaN.
    """
    return pd.read_csv(path, na_values="?", low_memory=False)
