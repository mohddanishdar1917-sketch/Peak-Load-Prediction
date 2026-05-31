import streamlit as st
import numpy as np
import joblib


#Page Config
st.set_page_config(

    page_title="Peak Load Prediction App",
    layout="centered"
)

#load trained objects
model = joblib.load("peak_load_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Peak Load Prediction")
st.markdown(
 """
    This application predicts the peak load for a given time period.
    Please enter the details in the sidebar.
    """
)

st.divider()
#---------- Sidebar: Inputs -----------

st.sidebar.header("Climate Information")

temperature = st.sidebar.number_input(
    "Temperature",
    min_value=-50,
    max_value=50,
    value=20,
    step=1
)

humidity = st.sidebar.number_input(
    "Humidity",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

wind_speed = st.sidebar.number_input(
    "Wind Speed",
    min_value=0,
    max_value=100,
    value=10,
    step=1
)

predict_button = st.sidebar.button("Predict")

if predict_button:
    data = np.array([[temperature, humidity, wind_speed]])
    data_scaled = scaler.transform(data)
    result = model.predict(data_scaled)[0]
    if result == 1:
        st.success("Peak load is expected to be high.")
    else:
        st.warning("Peak load is expected to be low.")
predict_button = st.sidebar.button("Predict")

if predict_button:
    data = np.array([[temperature, humidity, wind_speed]])
    data_scaled = scaler.transform(data)
    result = model.predict(data_scaled)[0]
    if result == 1:
        st.success("Peak load is expected to be high.")
    else:
        st.warning("Peak load is expected to be low.")