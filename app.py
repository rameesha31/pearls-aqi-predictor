import streamlit as st
import pandas as pd
import os
import joblib
import hopsworks
from dotenv import load_dotenv

load_dotenv()
print("CWD:", os.getcwd())
print("KEY FOUND:", repr(os.getenv("HOPSWORKS_API_KEY")))

st.set_page_config(page_title="AQI Dashboard", page_icon="🌫️", layout="wide")

# ---- Custom styling ----
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    * {
        color: #111827;
    }
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.7rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
    }
    h1, h2, h3, h4, h5 {
        color: #0f172a !important;
    }
    p, span, label, .stMarkdown {
        color: #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🌫️ AQI Forecast Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time air quality monitoring and 3-day forecasting for Lahore</p>', unsafe_allow_html=True)

@st.cache_data(ttl=1800)
def load_data():
    try:
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        fs = project.get_feature_store()
        fg = fs.get_feature_group("aqi_features", version=3)
        df = fg.read(read_options={"use_hive": True})
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df, "live"
    except Exception as e:
        # Fallback to local snapshot if Hopsworks is unavailable
        df = pd.read_csv("data_snapshot.csv", parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df, "cached"

@st.cache_resource
def load_models():
    try:
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        mr = project.get_model_registry()
        models = {}
        for label in ["24h", "48h", "72h"]:
            model_obj = mr.get_model(f"aqi_predictor_{label}", version=1)
            model_dir = model_obj.download()
            models[label] = joblib.load(os.path.join(model_dir, f"rf_{label}.pkl"))
        return models, "live"
    except Exception as e:
        # Fallback to local models if Hopsworks is unavailable
        models = {}
        for label in ["24h", "48h", "72h"]:
            models[label] = joblib.load(f"models/rf_{label}.pkl")
        return models, "cached"

def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#16a34a"
    elif aqi <= 100:
        return "Moderate", "#ca8a04"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ea580c"
    elif aqi <= 200:
        return "Unhealthy", "#dc2626"
    elif aqi <= 300:
        return "Very Unhealthy", "#9333ea"
    else:
        return "Hazardous", "#7f1d1d"

feature_cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2",
                 "hour", "day_of_week", "month",
                 "temperature", "humidity", "wind_speed"]

df, data_source = load_data()
models, model_source = load_models()

if data_source == "cached" or model_source == "cached":
    st.caption("⚠️ Showing cached data — live connection to Hopsworks is temporarily unavailable.")
else:
    st.caption("✅ Connected to live Hopsworks Feature Store")
latest = df.iloc[-1]
category, color = aqi_category(latest["aqi"])

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 3-Day Forecast", "📈 Historical Trends", "🧠 Model Insights"])

# ============ TAB 1: OVERVIEW ============
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current AQI", int(latest["aqi"]))
    col2.metric("PM2.5", f"{latest['pm2_5']:.1f}")
    col3.metric("Temperature", f"{latest['temperature']:.1f}°C")
    col4.metric("Humidity", f"{latest['humidity']:.0f}%")

    st.markdown(
        f"""<div style="padding: 1rem; border-radius: 0.6rem; background-color: {color}15;
        border-left: 5px solid {color}; margin-top: 0.5rem;">
        <h3 style="color: {color} !important; margin: 0;">Air Quality: {category}</h3></div>""",
        unsafe_allow_html=True
    )
    st.write("")
    if latest["aqi"] > 150:
        st.error(f"⚠️ Air quality is currently **{category}**. Consider limiting outdoor activity.")
    elif latest["aqi"] > 100:
        st.warning(f"⚠️ Air quality is **{category}**. Sensitive groups should take precautions.")
    else:
        st.success(f"✅ Air quality is **{category}**.")

    st.subheader("Current Pollutant Levels")
    pollutant_cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2"]
    st.bar_chart(latest[pollutant_cols], color="#2563eb")
    st.caption(f"Last updated: {latest['timestamp']}")

# ============ TAB 2: 3-DAY FORECAST ============
with tab2:
    st.subheader("Next 3 Days — AQI Forecast")

    latest_features = df[feature_cols].iloc[[-1]]

    cols = st.columns(3)
    labels = [("24h", "Day 1"), ("48h", "Day 2"), ("72h", "Day 3")]

    for col, (key, day_label) in zip(cols, labels):
        pred = models[key].predict(latest_features)[0]
        pred_cat, pred_color = aqi_category(pred)
        with col:
            st.markdown(f"#### {day_label}")
            st.markdown(
                f"""<div style="padding: 1.3rem; border-radius: 0.7rem; background-color: {pred_color}12;
                border: 1.5px solid {pred_color}55; text-align: center;">
                <h1 style="color: {pred_color} !important; margin: 0; font-size: 2.5rem;">{int(pred)}</h1>
                <p style="color: {pred_color} !important; margin: 0.3rem 0 0 0; font-weight: 700;">{pred_cat}</p>
                </div>""",
                unsafe_allow_html=True
            )

    st.info("Forecasts generated using Random Forest models trained separately for each time horizon (24h, 48h, 72h ahead), using historical pollutant and weather patterns.")

# ============ TAB 3: HISTORICAL TRENDS ============
with tab3:
    st.subheader("AQI Trend (Last 7 Days)")
    st.line_chart(df.set_index("timestamp")["aqi"].tail(168), color="#2563eb")

    st.subheader("Full Historical Trend")
    if os.path.exists("eda_outputs/1_aqi_trend.png"):
        col_a, col_b, col_c = st.columns([1, 3, 1])
        with col_b:
            st.image("eda_outputs/1_aqi_trend.png", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("By Hour of Day")
        if os.path.exists("eda_outputs/3_aqi_by_hour.png"):
            st.image("eda_outputs/3_aqi_by_hour.png", width=450)
    with col2:
        st.subheader("By Month (Seasonal)")
        if os.path.exists("eda_outputs/4_aqi_by_month.png"):
            st.image("eda_outputs/4_aqi_by_month.png", width=450)

    with st.expander("View raw data (last 20 readings)"):
        st.dataframe(df.tail(20))

# ============ TAB 4: MODEL INSIGHTS ============
with tab4:
    st.subheader("What Drives the AQI Prediction?")
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("eda_outputs/7_shap_importance_bar.png"):
            st.image("eda_outputs/7_shap_importance_bar.png", caption="Feature importance ranking", width=450)
    with col2:
        if os.path.exists("eda_outputs/6_shap_summary.png"):
            st.image("eda_outputs/6_shap_summary.png", caption="SHAP summary (impact direction)", width=450)

    st.subheader("Feature Correlation")
    if os.path.exists("eda_outputs/2_correlation_heatmap.png"):
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            st.image("eda_outputs/2_correlation_heatmap.png", use_container_width=True)

    st.subheader("PM2.5 vs AQI Relationship")
    if os.path.exists("eda_outputs/5_pm25_vs_aqi.png"):
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            st.image("eda_outputs/5_pm25_vs_aqi.png", use_container_width=True)
