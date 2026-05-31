# ===============================
# Peak Load Prediction Streamlit App (Minimal Inputs)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Peak Load Prediction", layout="centered")
st.title("⚡ Peak Load Prediction")
st.markdown("Accurate Forecasting Improves Energy Management")

# -------------------------------
# Generate Sample Training Data
# -------------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    rows = 500
    df = pd.DataFrame({
        "Temperature": np.random.uniform(0, 35, rows),
        "Humidity": np.random.uniform(30, 90, rows),
        "Wind_Speed": np.random.uniform(0, 20, rows),
        "Holiday_Flag": np.random.choice([0, 1], rows),
        "Power_Load_kw": np.random.uniform(200, 800, rows)
    })
    return df

df = generate_data()

# -------------------------------
# Train Model
# -------------------------------
X = df[["Temperature", "Humidity", "Wind_Speed", "Holiday_Flag"]]
y = df["Power_Load_kw"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.success(f"Model trained successfully! RMSE: {rmse:.2f}")

# -------------------------------
# User Inputs
# -------------------------------
st.header("🔮 Forecast Simulation")

temperature = st.slider("Temperature (°C)", 0, 40, 25)
humidity = st.slider("Humidity (%)", 20, 100, 60)
wind_speed = st.slider("Wind Speed (km/h)", 0, 25, 10)
holiday_flag = st.selectbox("Holiday?", [0, 1])

input_data = pd.DataFrame({
    "Temperature": [temperature],
    "Humidity": [humidity],
    "Wind_Speed": [wind_speed],
    "Holiday_Flag": [holiday_flag]
})

# -------------------------------
# Prediction
# -------------------------------
predicted_load = model.predict(input_data)[0]
st.info(f"🔺 Predicted Load: **{predicted_load:.2f} kWh**")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("**Developed by Mohammad Danish, Tahira Mohammad, and Fatima Banoo**")
