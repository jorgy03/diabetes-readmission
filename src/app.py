import numpy as np
import pandas as pd
import streamlit as st

from src.config import PROCESSED_DATA_DIR, TARGET_COLUMN
from src.models.predict import load_model, predict_proba

DIAG_CATEGORIES = [
    "diabetes", "circulatory", "respiratory", "digestive", "genitourinary",
    "injury", "musculoskeletal", "neoplasms", "other",
]
AGE_BUCKETS = [f"[{i}-{i + 10})" for i in range(0, 100, 10)]

st.set_page_config(page_title="Diabetes Readmission Risk", page_icon="🏥")
st.title("30-Day Readmission Risk")
st.caption(
    "Estimates the risk that a diabetic patient is readmitted to the hospital "
    "within 30 days, using a Gradient Boosting model trained on the UCI "
    "Diabetes 130-US Hospitals dataset. For demo purposes only — not a "
    "clinical tool."
)


@st.cache_resource
def get_model():
    return load_model("best_model.joblib")


@st.cache_data
def get_feature_template():
    """All columns the model was trained on, as a one-row all-NaN template —
    fields the user doesn't fill in get median/mode-imputed by the pipeline.

    object dtype throughout: a plain float64 NaN column raises a TypeError
    the moment a string (e.g. age bucket) is assigned into it.
    """
    df = pd.read_csv(PROCESSED_DATA_DIR / "diabetic_data_clean.csv")
    columns = [c for c in df.columns if c != TARGET_COLUMN]
    return pd.DataFrame([{c: np.nan for c in columns}], dtype=object)


model = get_model()
template = get_feature_template()

st.subheader("Patient & encounter details")
col1, col2 = st.columns(2)

with col1:
    age = st.selectbox("Age group", AGE_BUCKETS, index=6)
    time_in_hospital = st.slider("Days in hospital (this stay)", 1, 14, 3)
    num_medications = st.slider("Number of medications", 1, 40, 12)
    number_diagnoses = st.slider("Number of diagnoses", 1, 16, 7)
    num_lab_procedures = st.slider("Number of lab procedures", 1, 100, 40)

with col2:
    number_inpatient = st.slider("Prior inpatient visits (past year)", 0, 10, 0)
    number_emergency = st.slider("Prior emergency visits (past year)", 0, 10, 0)
    number_outpatient = st.slider("Prior outpatient visits (past year)", 0, 10, 0)
    insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
    diabetes_med = st.selectbox("On a diabetes medication?", ["Yes", "No"])

st.subheader("Primary diagnosis")
diag_1_cat = st.selectbox("Primary diagnosis category", DIAG_CATEGORIES)

with st.expander("More fields (optional — left blank fields use typical values)"):
    change = st.selectbox("Medication changed this encounter?", ["No", "Ch"], index=0)
    a1c_result = st.selectbox("A1C result", [None, "Norm", ">7", ">8"], index=0)
    max_glu_serum = st.selectbox("Max glucose serum", [None, "Norm", ">200", ">300"], index=0)
    race = st.selectbox("Race", [None, "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"], index=0)
    gender = st.selectbox("Gender", [None, "Female", "Male"], index=0)

if st.button("Estimate risk", type="primary"):
    row = template.copy()
    row.loc[0, "age"] = age
    row.loc[0, "time_in_hospital"] = time_in_hospital
    row.loc[0, "num_medications"] = num_medications
    row.loc[0, "number_diagnoses"] = number_diagnoses
    row.loc[0, "num_lab_procedures"] = num_lab_procedures
    row.loc[0, "number_inpatient"] = number_inpatient
    row.loc[0, "number_emergency"] = number_emergency
    row.loc[0, "number_outpatient"] = number_outpatient
    row.loc[0, "insulin"] = insulin
    row.loc[0, "diabetesMed"] = diabetes_med
    row.loc[0, "diag_1_cat"] = diag_1_cat
    row.loc[0, "change"] = change
    row.loc[0, "A1Cresult"] = a1c_result
    row.loc[0, "max_glu_serum"] = max_glu_serum
    row.loc[0, "race"] = race
    row.loc[0, "gender"] = gender

    risk = predict_proba(model, row)[0]

    st.metric("Estimated 30-day readmission risk", f"{risk:.1%}")
    if risk >= 0.5:
        st.warning("Above the model's decision threshold — flagged as higher risk.")
    else:
        st.success("Below the model's decision threshold — flagged as lower risk.")
    st.caption(
        "The dataset's actual readmission rate is ~11%, so anything meaningfully "
        "above that is elevated risk relative to the population."
    )
