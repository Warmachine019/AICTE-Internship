# Importing Python Libraries
import streamlit as st
import joblib
import pandas as pd
import os

# Loading the model, scaler and the features
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
features_path = os.path.join(BASE_DIR, "models", "feature_columns.pkl")


model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_cols = joblib.load(features_path)

st.title("🌿 Supply Chain Emissions Predictor")

st.markdown("""
Estimate **Supply Chain Emission Factors with Margins** using key supply chain parameters and data quality indicators.
""")

with st.form("prediction_form"):
    st.selectbox("Select Substance", [
        'Carbon Dioxide', 'Methane', 'Nitrous Oxide', 'Other GHGs'
    ])
    st.selectbox("Select Unit", [
        'kg/2018 USD, purchaser price',
        'kg CO2e/2018 USD, purchaser price'
    ])
    st.selectbox("Data Source", [
        'Commodity', 'Industry'
    ])

    supply_wo_margin = st.number_input(
        "Emission Factor without Margins", min_value=0.0)
    margin = st.number_input(
        "Margin of Emission Factor", min_value=0.0)

    dq_reliability = st.slider(
        "DQ Reliability Score", 0.0, 5.0, step=1.0)
    dq_temporal = st.slider(
        "DQ Temporal Correlation", 0.0, 5.0, step=1.0)
    dq_geo = st.slider(
        "DQ Geographical Correlation", 0.0, 5.0, step=1.0)
    dq_tech = st.slider(
        "DQ Technological Correlation", 0.0, 5.0, step=1.0)
    dq_data = st.slider(
        "DQ Data Collection Quality", 0.0, 5.0, step=1.0)

    submit = st.form_submit_button("Predict Emission Factor")

# Make Prediction
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

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_cols]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"🎯 Estimated Emission Factor with Margin: **{prediction[0]:.4f}**")

# Footer
st.markdown(
    """
    <hr style="margin-top:50px;">
    <div style="text-align: center; color: gray;">
        Made by Spandan
    </div>
    """,
    unsafe_allow_html=True
)