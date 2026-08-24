import streamlit as st
import hopsworks
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AQI Dashboard", page_icon="🌫️")
st.title("🌫️ AQI Forecast Dashboard — Lahore")

# Connect to Hopsworks
@st.cache_resource
def get_feature_store():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    return fs

# Load data (cached so it doesn't reload every time)
@st.cache_data(ttl=600)  # refresh every 10 minutes
def load_data():
    fs = get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read(read_options={"use_hive": True})
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# Load and show data
with st.spinner("Loading latest data..."):
    df = load_data()

latest = df.iloc[-1]

st.subheader("Current Readings")
col1, col2, col3 = st.columns(3)
col1.metric("Current AQI", int(latest["aqi"]))
col2.metric("PM2.5", f"{latest['pm2_5']:.1f}")
col3.metric("Temperature", f"{latest['temperature']:.1f}°C")

st.subheader("Recent AQI Trend")
st.line_chart(df.set_index("timestamp")["aqi"].tail(48))

st.subheader("Raw Data (last 10 readings)")
st.dataframe(df.tail(10))