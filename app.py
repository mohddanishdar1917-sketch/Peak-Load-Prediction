# ===============================
# Peak Load Prediction Streamlit App
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Peak Load Prediction", layout="wide")
st.title("⚡ Peak Load Prediction Dashboard")
st.markdown("Accurate Forecasting Improves Energy Management")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Upload Datasets")
power_file = st.sidebar.file_uploader("Upload Household Power Data (CSV)", type=["csv"])
temp_file = st.sidebar.file_uploader("Upload Temperature Data (CSV)", type=["csv"])
humidity_file = st.sidebar.file_uploader("Upload Humidity Data (CSV)", type=["csv"])
wind_file = st.sidebar.file_uploader("Upload Wind Speed Data (CSV)", type=["csv"])
holiday_file = st.sidebar.file_uploader("Upload Holiday Flag Data (CSV)", type=["csv"])

# -------------------------------
# Main Workflow
# -------------------------------
if st.sidebar.button("Run Prediction"):
    if not all([power_file, temp_file, humidity_file, wind_file, holiday_file]):
        st.error("Please upload all required datasets.")
    else:
        # Load datasets
        df_power = pd.read_csv(power_file)
        df_temp = pd.read_csv(temp_file)
        df_humidity = pd.read_csv(humidity_file)
        df_wind = pd.read_csv(wind_file)
        df_holiday = pd.read_csv(holiday_file)

        # Combine Date and Time
        df_power['datetime'] = df_power['Date'].astype(str).str.strip() + ' ' + df_power['Time'].astype(str).str.strip()
        df_power = df_power.drop(columns=['Date', 'Time'])
        df_power['datetime'] = pd.to_datetime(df_power['datetime'], dayfirst=True, errors='coerce')

        # Convert datetime columns
        for df in [df_temp, df_humidity, df_wind, df_holiday]:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df['datetime'] = df['datetime'].dt.floor('h')

        # Rename columns
        df_temp = df_temp[['datetime', 'Chicago']].rename(columns={'Chicago': 'Temperature'})
        df_humidity = df_humidity[['datetime', 'Chicago']].rename(columns={'Chicago': 'Humidity'})
        df_wind = df_wind[['datetime', 'Chicago']].rename(columns={'Chicago': 'Wind_Speed'})

        # Merge datasets
        df = df_power.merge(df_temp, on='datetime', how='left')
        df = df.merge(df_humidity, on='datetime', how='left')
        df = df.merge(df_wind, on='datetime', how='left')
        df = df.merge(df_holiday, on='datetime', how='left')

        df = df.replace('?', np.nan)
        for col in df.columns:
            if col != 'datetime':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.ffill().bfill()
        df['Holiday_Flag'] = df['Holiday_Flag'].fillna(0)

        # Feature Engineering
        df['Hour'] = df['datetime'].dt.hour
        df['Day'] = df['datetime'].dt.day
        df['Month'] = df['datetime'].dt.month
        df['Weekday'] = df['datetime'].dt.weekday

        # Define features and target
        X = df[['Hour','Day','Month','Weekday','Temperature','Humidity','Wind_Speed','Holiday_Flag']]
        y = df['Power_Load_kw']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        # Model training
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.success(f"Model trained successfully! RMSE: {rmse:.2f}")

        # Visualization
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(y_test.values, label="Actual Demand", color="blue")
        ax.plot(y_pred, label="Predicted Demand", color="red")
        ax.set_title("Actual vs Predicted Electricity Demand")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Demand (kWh)")
        ax.legend()
        st.pyplot(fig)

        # Peak Load Prediction
        peak_index = np.argmax(y_pred)
        actual_test_row_index = len(X_train) + peak_index
        predicted_timestamp = df.iloc[actual_test_row_index]['datetime']
        st.info(f"🔺 Predicted Peak Load Time: {predicted_timestamp}")
        st.info(f"🔺 Predicted Peak Demand: {y_pred[peak_index]:.2f} kWh")

        # Save model
        joblib.dump(model, 'peak_load_model.pkl')
        st.success("Model saved as 'peak_load_model.pkl'")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("**Developed by Mohammad Danish, Tahira Mohammad, and Fatima Banoo**")
