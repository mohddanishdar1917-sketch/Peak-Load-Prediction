import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Peak Load Prediction Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.title("⚡ Peak Load Prediction Dashboard")
st.markdown(
    "### Accurate Forecasting Improves Energy Management"
)

st.markdown("---")

# ==========================================
# GENERATE SAMPLE DATA
# ==========================================

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

# ==========================================
# MODEL TRAINING
# ==========================================

X = df[[
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Holiday_Flag"
]]

y = df["Power_Load_kw"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    shuffle=False
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("⚙️ Input Parameters")

temperature = st.sidebar.slider(
    "Temperature (°C)",
    0,
    40,
    25
)

humidity = st.sidebar.slider(
    "Humidity (%)",
    20,
    100,
    60
)

wind_speed = st.sidebar.slider(
    "Wind Speed (km/h)",
    0,
    25,
    10
)

holiday_option = st.sidebar.selectbox(
    "Holiday?",
    ["No", "Yes"]
)

holiday_flag = 1 if holiday_option == "Yes" else 0

# ==========================================
# KPI CARDS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🌡 Temperature",
        f"{temperature} °C"
    )

with col2:
    st.metric(
        "💧 Humidity",
        f"{humidity}%"
    )

with col3:
    st.metric(
        "💨 Wind Speed",
        f"{wind_speed} km/h"
    )

with col4:
    st.metric(
        "🎉 Holiday",
        holiday_option
    )

st.markdown("---")

# ==========================================
# PREDICTION SECTION
# ==========================================

st.subheader("🔮 Forecast Simulation")

input_data = pd.DataFrame({
    "Temperature": [temperature],
    "Humidity": [humidity],
    "Wind_Speed": [wind_speed],
    "Holiday_Flag": [holiday_flag]
})

if st.button("🚀 Predict Peak Load"):

    predicted_load = model.predict(input_data)[0]

    st.success("Prediction Generated Successfully")

    # KPI Result

    st.metric(
        label="⚡ Predicted Peak Load",
        value=f"{predicted_load:.2f} kWh"
    )

    # Gauge Chart

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=predicted_load,
            title={"text": "Predicted Load (kWh)"},
            gauge={
                "axis": {"range": [0, 1000]}
            }
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # Progress Bar

    percentage = min(
        predicted_load / 1000,
        1.0
    )

    st.progress(float(percentage))

    # Download Result

    result_df = pd.DataFrame({
        "Predicted_Load_kWh": [predicted_load]
    })

    st.download_button(
        label="📥 Download Prediction",
        data=result_df.to_csv(index=False),
        file_name="prediction.csv",
        mime="text/csv"
    )

st.markdown("---")

# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.subheader("📊 Model Performance")

st.metric(
    "RMSE",
    f"{rmse:.2f}"
)

st.markdown("---")

# ==========================================
# ANALYTICS TABS
# ==========================================

tab1, tab2 = st.tabs([
    "📈 Analytics",
    "📋 Dataset Preview"
])

with tab1:

    st.subheader(
        "Historical Power Consumption"
    )

    fig = px.line(
        df,
        y="Power_Load_kw",
        title="Power Load Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab2:

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

st.markdown("---")

# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer">
    Developed by Mohammad Danish, Tahira Mohammad, and Fatima Banoo
    </div>
    """,
    unsafe_allow_html=True
)