import streamlit as st
st.set_page_config(page_title="Pipe Lifetime Predictor", layout="centered")

import pandas as pd
import joblib
import numpy as np
from pathlib import Path


# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    # pipe_lifetime_app.py is in MLAdditive/Simulated-Software/
    # model is in MLAdditive/models/
    root = Path(__file__).resolve().parents[1]  # ==> MLAdditive/
    model_path = root / "models" / "best_rf_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"CWD: {Path.cwd()}\n"
            f"Here dir: {Path(__file__).resolve().parent}\n"
            f"PKLs I can see from root: {[str(p) for p in root.glob('**/*.pkl')]}"
        )
    return joblib.load(model_path)

model = load_model()

# Try to resolve expected columns file relative to repo
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COLS_PATH = ROOT / "data" / "expected_columns.txt"

def _load_expected_columns():
    if EXPECTED_COLS_PATH.exists():
        with open(EXPECTED_COLS_PATH, "r") as f:
            return [line.strip() for line in f if line.strip()]
    # Fallback: scikit-learn >=1.0 usually exposes this
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    raise FileNotFoundError(
        f"Could not find {EXPECTED_COLS_PATH} and model has no feature_names_in_. "
        "Please commit expected_columns.txt or upgrade sklearn / persist feature names."
    )

EXPECTED_COLS = _load_expected_columns()

# ------------------ Preprocessing ------------------
@st.cache_data
def preprocess(df):
    X = df.drop('Lifetime_years', axis=1, errors='ignore')
    y = df['Lifetime_years'] if 'Lifetime_years' in df else None

    # Encode categorical columns
    cat_cols = [
        'Base_Resin', 'SDR', 'Environment', 'Primary_AO', 'Secondary_AO',
        'Carbon_Black_Type', 'Wax_Type'
    ]
    X = pd.get_dummies(X, columns=[c for c in cat_cols if c in X.columns])

    # Ensure all expected columns exist and in the right order
    for col in EXPECTED_COLS:
        if col not in X.columns:
            X[col] = 0
    X = X[EXPECTED_COLS]

    return X, y

# ------------------ Streamlit App ------------------
st.title("Pipe Lifetime Prediction App")
st.markdown("Enter the properties of the pipe to predict its estimated **Lifetime (years)**.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        base_resin = st.selectbox("Base Resin", ['PE100', 'PE100-RC'])
        sdr = st.selectbox("SDR", ['6', '9', '11', '17', '21', '26'])
        pressure = st.slider("Service Pressure (bar)", 0, 25, 10)
        temp = st.slider("Service Temperature (°C)", 0, 100, 23)
        env = st.selectbox("Environment", ['Water', 'Soil', 'Air', 'ClO2'])
        primary_ao = st.selectbox("Primary AO", ['Irganox 1010', 'Irganox 1076', 'Hostanox O3'])
        primary_ao_ppm = st.number_input("Primary AO (ppm)", 0.0, 5000.0, 925.0)

    with col2:
        secondary_ao = st.selectbox("Secondary AO", ['Irgafos 168', 'Sandostab PEPQ', 'PEPQ'])
        secondary_ao_ppm = st.number_input("Secondary AO (ppm)", 0.0, 5000.0, 1200.0)
        cb_type = st.selectbox("Carbon Black Type", ['Standard CB', 'Conductive CB', 'High-Structure CB', 'None'])
        cb_content = st.slider("CB Content (%)", 0.0, 10.0, 1.0)
        wax_type = st.selectbox("Wax Type", ['Erucamide', 'Amide Wax', 'None', 'Oxidized PE Wax'])
        wax_content = st.slider("Wax Content (%)", 0.0, 5.0, 1.0)

    submitted = st.form_submit_button("Predict Lifetime")

if submitted:
    user_input = pd.DataFrame([{
        "Base_Resin": base_resin,
        "SDR": sdr,
        "Service_Pressure_bar": pressure,
        "Service_Temperature_C": temp,
        "Environment": env,
        "Primary_AO": primary_ao,
        "Secondary_AO": secondary_ao,
        "Primary_AO_ppm": primary_ao_ppm,
        "Secondary_AO_ppm": secondary_ao_ppm,
        "Carbon_Black_Type": cb_type,
        "CB_Content_%": cb_content,
        "Wax_Type": wax_type,
        "Wax_Content_%": wax_content,
        "Lifetime_years": 0  # placeholder
    }])

    X_transformed, _ = preprocess(user_input)
    prediction = float(model.predict(X_transformed)[0])

    st.success(f"Predicted Lifetime: **{prediction:.2f} years**")

    # Clamp 0..100 for progress bar
    pct = int(np.clip(prediction * 2, 0, 100))
    st.progress(pct)