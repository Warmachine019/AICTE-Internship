import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ------------------------------
# Load model, scaler, and features
# ------------------------------
model_path = Path("models/random_forest_model.pkl")
scaler_path = Path("models/scaler.pkl")
features_path = Path("models/feature_columns.pkl")


model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_cols = joblib.load(features_path)  # Ensure correct input order

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Supply Chain Emissions Prediction")

st.markdown("""
This app predicts **Supply Chain Emission Factors with Margins**  
based on supply chain parameters and DQ (Data Quality) metrics.
""")

with st.form("prediction_form"):
    # Unused dropdowns (if not part of model)
    st.selectbox("Substance", ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'])
    st.selectbox("Unit", ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'])
    st.selectbox("Source", ['Commodity', 'Industry'])

    # Input features
    supply_wo_margin = st.number_input("Supply Chain Emission Factors without Margins", min_value=0.0)
    margin = st.number_input("Margins of Supply Chain Emission Factors", min_value=0.0)
    dq_reliability = st.slider("DQ Reliability", 0.0, 5.0, step=1.0)
    dq_temporal = st.slider("DQ Temporal Correlation", 0.0, 5.0, step=1.0)
    dq_geo = st.slider("DQ Geographical Correlation", 0.0, 5.0, step=1.0)
    dq_tech = st.slider("DQ Technological Correlation", 0.0, 5.0, step=1.0)
    dq_data = st.slider("DQ Data Collection", 0.0, 5.0, step=1.0)

    submit = st.form_submit_button("Predict")

# ------------------------------
# Make prediction
# ------------------------------
if submit:
    input_data = {
        'Supply Chain Emission Factors without Margins': supply_wo_margin,
        'Margins of Supply Chain Emission Factors': margin,
        'DQ ReliabilityScore of Factors without Margins': dq_reliability,
        'DQ TemporalCorrelation of Factors without Margins': dq_temporal,
        'DQ GeographicalCorrelation of Factors without Margins': dq_geo,
        'DQ TechnologicalCorrelation of Factors without Margins': dq_tech,
        'DQ DataCollection of Factors without Margins': dq_data
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure correct feature order
    input_df = input_df[feature_cols]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"🎯 Predicted Supply Chain Emission Factor with Margin: **{prediction[0]:.4f}**")
