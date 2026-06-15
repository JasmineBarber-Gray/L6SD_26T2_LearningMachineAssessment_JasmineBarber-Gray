import streamlit as st
import numpy as np
import joblib

# load model and scaler
model = joblib.load("model/networth_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Title and description
st.title("Net Worth Predictor")
st.write("Enter financial details below to predict net worth")

# input fields
age = st.number_input("Age (min: 18, max: 100)", min_value=18, max_value=100, value=30)
income = st.number_input("Income (min: 0, max: 1,000,000)", min_value=0, max_value=1_000_000, value=50000)
debt = st.number_input("Credit Card Debt (min: 0, max: 500,000)", min_value=0, max_value=500_000, value=10000)
stocks = st.number_input("Stocks (min: 0.0, max: 1,000,000.0)", min_value=0.0, max_value=1_000_000.0, value=20000.0)
mutual_funds = st.number_input("Mutual Funds (min: 0.0, max: 1,000,000.0)", min_value=0.0, max_value=1_000_000.0, value=15000.0)

# predict button
if st.button("Predict Net Worth"):

    # input must match training order EXACTLY
    input_data = np.array([[age, income, debt, stocks, mutual_funds]])

    # scale (IMPORTANT for linear regression)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.success(f"Estimated Net Worth: ${prediction:,.2f}")

    # optional interpretation
    st.info("Prediction is based on trained ML model using historical financial data. Actual net worth may vary based on many factors.")