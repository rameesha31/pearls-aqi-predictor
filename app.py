import streamlit as st
import hopsworks
import pandas as pd
import os
import joblib
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AQI Dashboard", page_icon="🌫️", layout="wide")
st.title("🌫️ AQI Forecast Dashboard — Lahore")

# ---- Connect to Hopsworks (cached so it only connects once) ----
@st.cache_resource
def get_project():
    return hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))

# ---- Load feature data ----
@st.cache_data(ttl=600)  # refresh every 10 minutes
def load_data():
    project = get_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=3)
    df = fg.read(read_options={"use_hive": True})
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# ---- Load trained model from Hopsworks Model Registry ----
@st.cache_resource
def load_model():
    project = get_project()
    mr = project.get_model_registry()
    model_obj = mr.get_model("aqi_predictor", version=1)
    model_dir = model_obj.download()
    model = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
    return model

# ---- Helper: convert AQI number to category + color ----
def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "orange"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    elif aqi <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "darkred"

# ---- Load data ----
with st.spinner("Loading latest data..."):
    df = load_data()

latest = df.iloc[-1]
category, color = aqi_category(latest["aqi"])

# ---- Top metrics ----
st.subheader("Current Readings")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current AQI", int(latest["aqi"]))
col2.metric("PM2.5", f"{latest['pm2_5']:.1f}")
col3.metric("Temperature", f"{latest['temperature']:.1f}°C")
col4.metric("Humidity", f"{latest['humidity']:.0f}%")

st.markdown(f"### Air Quality Status: :{color}[{category}]")

# ---- Alert for hazardous levels ----
if latest["aqi"] > 150:
    st.error(f"⚠️ Air quality is currently **{category}**. Consider limiting outdoor activity.")
elif latest["aqi"] > 100:
    st.warning(f"⚠️ Air quality is **{category}**. Sensitive groups should take precautions.")
else:
    st.success(f"✅ Air quality is **{category}**.")

# ---- Make prediction ----
with st.spinner("Loading model and generating forecast..."):
    model = load_model()

feature_cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2",
                 "hour", "day_of_week", "month",
                 "temperature", "humidity", "wind_speed"]

latest_features = df[feature_cols].iloc[[-1]]  # last row, as a DataFrame
predicted_aqi = model.predict(latest_features)[0]
pred_category, pred_color = aqi_category(predicted_aqi)

st.subheader("24-Hour Forecast")
col_a, col_b = st.columns(2)
col_a.metric("Predicted AQI (24h ahead)", int(predicted_aqi))
col_b.markdown(f"### Status: :{pred_color}[{pred_category}]")

# ---- Trend chart ----
st.subheader("Recent AQI Trend (last 7 days)")
st.line_chart(df.set_index("timestamp")["aqi"].tail(168))

# ---- Pollutant breakdown ----
st.subheader("Current Pollutant Levels")
pollutant_cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2"]
pollutant_data = latest[pollutant_cols]
st.bar_chart(pollutant_data)

# ---- Raw data table ----
with st.expander("View raw data (last 20 readings)"):
    st.dataframe(df.tail(20))

st.caption(f"Last updated: {latest['timestamp']}")