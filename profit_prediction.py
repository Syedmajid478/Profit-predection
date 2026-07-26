import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Profit Prediction",
    page_icon="📈"
)
# Load Model and Encoder
rf_model = joblib.load("random_model.pkl")
state_encoder = joblib.load("state_encoder.pkl")

st.title("📈 Company Profit Prediction")
st.write("Predict company profit using Machine Learning (Random Forest).")

# User Inputs
rd_spend = st.number_input("💻 R&D Spend", min_value=0.0)

administration = st.number_input("🏢 Administration", min_value=0.0)

marketing_spend = st.number_input("📢 Marketing Spend", min_value=0.0)

state = st.selectbox(
    "🌍 Select State",
    state_encoder.classes_
)

# Predict Button
if st.button("Predict Profit 💰"):

    state_value = state_encoder.transform([state])[0]

    test = pd.DataFrame({
        "R&D Spend": [rd_spend],
        "Administration": [administration],
        "Marketing Spend": [marketing_spend],
        "State": [state_value]
    })

    prediction = rf_model.predict(test)

    st.success(f"💰 Predicted Profit: ₹ {prediction[0]:,.2f}")