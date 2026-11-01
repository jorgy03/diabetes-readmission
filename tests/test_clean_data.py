import pandas as pd

from src.data.clean_data import (
    add_diagnosis_categories,
    add_readmission_target,
    categorize_diagnosis,
    drop_duplicate_patients,
    drop_expired_patients,
)


def test_add_readmission_target_is_binary():
    df = pd.DataFrame({"readmitted": ["<30", ">30", "NO"]})
    result = add_readmission_target(df)
    assert result["readmitted_30d"].tolist() == [1, 0, 0]
    assert "readmitted" not in result.columns


def test_drop_duplicate_patients_keeps_first_encounter():
    df = pd.DataFrame({
        "patient_nbr": [1, 1, 2],
        "encounter_order": [1, 2, 1],
    })
    result = drop_duplicate_patients(df)
    assert len(result) == 2
    assert result[result["patient_nbr"] == 1]["encounter_order"].iloc[0] == 1


def test_categorize_diagnosis_buckets_known_codes():
    assert categorize_diagnosis("250.83") == "diabetes"
    assert categorize_diagnosis("410") == "circulatory"
    assert categorize_diagnosis("786") == "respiratory"
    assert categorize_diagnosis("V27") == "other"
    assert categorize_diagnosis(None) == "other"


def test_add_diagnosis_categories_drops_raw_columns():
    df = pd.DataFrame({"diag_1": ["250.83"], "diag_2": ["410"], "diag_3": ["V27"]})
    result = add_diagnosis_categories(df)
    assert list(result.columns) == ["diag_1_cat", "diag_2_cat", "diag_3_cat"]
    assert result.iloc[0].tolist() == ["diabetes", "circulatory", "other"]


def test_drop_expired_patients_removes_death_discharge_codes():
    df = pd.DataFrame({
        "discharge_disposition_id": [1, 11, 19, 20, 21, 3],
    })
    result = drop_expired_patients(df)
    assert result["discharge_disposition_id"].tolist() == [1, 3]
