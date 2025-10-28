import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app title
st.title("🏭 Manufacturing Output Prediction App")

st.write("Enter the input parameters below to predict the Parts Per Hour:")

# Input fields for user data
machine_speed = st.number_input("Machine Speed (rpm)", min_value=0.0)
material_density = st.number_input("Material Density (g/cm³)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
pressure = st.number_input("Pressure (bar)", min_value=0.0)

if st.button("Predict Output"):
    # Prepare input
    input_data = np.array([[machine_speed, material_density, temperature, pressure]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Parts Per Hour: {prediction[0]:.2f}")

st.caption("Built with ❤️ using Streamlit and Scikit-learn")
