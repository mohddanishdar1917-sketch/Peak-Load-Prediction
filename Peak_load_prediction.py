# ===============================
# Peak Load Prediction Project (Beginner Friendly)
# ===============================

# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

# -------------------------------
# 2. Load Datasets
# -------------------------------
# Replace with your actual file names
df_power = pd.read_csv("household_power_consumption.csv")
# Combine Date and Time into a single datetime column
# 1. Combine Date and Time strings cleanly
df_power['datetime'] = df_power['Date'].astype(str).str.strip() + ' ' + df_power['Time'].astype(str).str.strip()
df_power = df_power.drop(columns=['Date', 'Time'])

# 2. Convert df_power using dayfirst=True (standard for household power data formats)
df_power['datetime'] = pd.to_datetime(df_power['datetime'], dayfirst=True, errors='coerce')

# 3. Process the other datasets normally down to the hour

df_temp = pd.read_csv("temperature.csv")
df_humidity = pd.read_csv("humidity.csv")
df_wind = pd.read_csv("wind_speed.csv")
df_holiday = pd.read_csv("holiday_flag.csv")

# -------------------------------
# 3. Convert DateTime Columns
# -------------------------------
print("Power cols:", df_power.columns.tolist())
print("Temperature cols:", df_temp.columns.tolist())
print("Humidity cols:", df_humidity.columns.tolist())
print("Wind cols:", df_wind.columns.tolist())
print("Holiday cols:", df_holiday.columns.tolist())

for df in [df_temp, df_humidity, df_wind, df_holiday]:
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed', errors='coerce')

# 4. Align all datasets to the identical hourly format so they match perfectly
for df in [df_power, df_temp, df_humidity, df_wind, df_holiday]:
    df['datetime'] = df['datetime'].dt.floor('h')


# -------------------------------
# 4. Merge All Datasets
# -------------------------------
# 1. Clean up and aggregate datasets to ensure exactly ONE row per datetime
# 1. Clean up and aggregate datasets to ensure exactly ONE row per datetime
# 1. Select a specific city and rename the column so your features match later
# (Change 'Chicago' to whichever city you want to predict for)
df_temp = df_temp[['datetime', 'Chicago']].rename(columns={'Chicago': 'Temperature'})
df_humidity = df_humidity[['datetime', 'Chicago']].rename(columns={'Chicago': 'Humidity'})
df_wind = df_wind[['datetime', 'Chicago']].rename(columns={'Chicago': 'Wind_Speed'})

# Group by datetime to handle any duplicates safely
df_temp = df_temp.groupby('datetime').mean().reset_index()
df_humidity = df_humidity.groupby('datetime').mean().reset_index()
df_wind = df_wind.groupby('datetime').mean().reset_index()

# For the holiday dataset, keep it as is
df_holiday['datetime'] = pd.to_datetime(df_holiday['datetime'], format='mixed', errors='coerce')
df_holiday = df_holiday.drop_duplicates(subset=['datetime'])

df = df_power.merge(df_temp, on='datetime', how='left')
df = df.merge(df_humidity, on='datetime', how='left')
df = df.merge(df_wind, on='datetime', how='left')
df = df.merge(df_holiday, on='datetime', how='left')

df = df.replace('?', np.nan)  # replace '?' with NaN
for col in df.columns:
    if col != 'datetime':
        df[col] = pd.to_numeric(df[col], errors='coerce')  # convert to numeric, coercing errors to NaN

# -------------------------------
# 5. Basic Cleaning
# -------------------------------
df = df.ffill().bfill()  # forward and backward fill to handle any remaining missing values
df['Holiday_Flag'] = df['Holiday_Flag'].fillna(0)  # fill missing holiday flags with 0 (not a holiday)

# -------------------------------
# 6. Feature Engineering
# -------------------------------
df['Hour'] = df['datetime'].dt.hour
df['Day'] = df['datetime'].dt.day
df['Month'] = df['datetime'].dt.month
df['Weekday'] = df['datetime'].dt.weekday

# -------------------------------
# 7. Define Features & Target
# -------------------------------
X = df[['Hour','Day','Month','Weekday',
        'Temperature','Humidity','Wind_Speed','Holiday_Flag']]
y = df['Power_Load_kw']   # target column from power dataset

# -------------------------------
# 8. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# -------------------------------
# 9. Model Training
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 10. Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 11. Evaluation
# -------------------------------
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

# -------------------------------
# 12. Visualization
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Demand", color="blue")
plt.plot(y_pred, label="Predicted Demand", color="red")
plt.title("Actual vs Predicted Electricity Demand")
plt.xlabel("Time Steps")
plt.ylabel("Demand (kWh)")
plt.legend()
plt.show()

# -------------------------------
# 13. Peak Load Prediction
# -------------------------------
peak_index = y_pred.argmax()
actual_test_row_index = len(X_train) + peak_index  # Get the actual index in the original dataframe for the predicted peak load
predicted_timestamp = df.iloc[actual_test_row_index]['datetime']
print("Predicted Peak Load Time:", predicted_timestamp)
print("Predicted Peak Demand:", y_pred[peak_index])



import joblib
# Save the model to a file
joblib.dump(model, 'peak_load_model.pkl')