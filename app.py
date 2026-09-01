import streamlit as st
import pandas as pd
import os
import joblib
import hopsworks
from dotenv import load_dotenv

load_dotenv()

USE_LIVE_HOPSWORKS = True 
st.set_page_config(page_title="AQI Dashboard", page_icon="🌫️", layout="wide")

# ---- Custom styling ----
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #ffffff;
    }

    /* ---- Force the built-in Streamlit top toolbar to white ---- */
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    div[data-testid="stToolbar"] {
        background-color: #ffffff !important;
    }
    div[data-testid="stDecoration"] {
        background: #ffffff !important;
    }

    * {
        color: #111827;
    }

    /* ---- Header banner ---- */
    .header-banner {
        background: linear-gradient(135deg, #0ea5e9 0%, #14b8a6 50%, #22c55e 100%);
        border-radius: 1.1rem;
        padding: 1.8rem 2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.25);
    }
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #f1f5f9 !important;
        font-size: 1.05rem;
        margin-bottom: 0;
        font-weight: 500;
    }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, #ffffff 0%, #ecfeff 100%);
        border: 1px solid #a5f3fc;
        border-left: 5px solid #0ea5e9;
        border-radius: 0.9rem;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 2px 10px rgba(14, 165, 233, 0.08);
        transition: transform 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(14, 165, 233, 0.15);
    }
    div[data-testid="stMetricLabel"] {
        color: #0369a1 !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
    }
    div[data-testid="stMetricValue"] {
        color: #1e1b4b !important;
        font-family: 'Poppins', sans-serif;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Poppins', sans-serif;
        color: #1e1b4b !important;
    }
    p, span, label, .stMarkdown {
        color: #1e293b;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #6b7280;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #0ea5e9 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #0ea5e9 !important;
    }

    /* ---- Expander & misc containers ---- */
    div[data-testid="stExpander"] {
        border: 1px solid #ede9fe;
        border-radius: 0.8rem;
        overflow: hidden;
    }

    /* ---- Alert boxes get rounder corners ---- */
    div[data-testid="stAlert"] {
        border-radius: 0.8rem;
    }

    hr {
        border-color: #ede9fe;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="header-banner">
        <p class="main-header">🌫️ AQI Forecast Dashboard</p>
        <p class="sub-header">Real-time air quality monitoring and 3-day forecasting for Lahore</p>
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_data(ttl=1800)
def load_data():
    if USE_LIVE_HOPSWORKS:
        try:
            project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
            fs = project.get_feature_store()
            fg = fs.get_feature_group("aqi_features", version=3)
            df = fg.read(read_options={"use_hive": True})
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df, "live"
        except Exception:
            pass  # fall through to cached version

    df = pd.read_csv("data_snapshot.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, "cached"

@st.cache_resource
def load_models():
    if USE_LIVE_HOPSWORKS:
        try:
            project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
            mr = project.get_model_registry()
            models = {}
            for label in ["24h", "48h", "72h"]:
                model_obj = mr.get_model(f"aqi_predictor_{label}", version=1)
                model_dir = model_obj.download()
                models[label] = joblib.load(os.path.join(model_dir, f"rf_{label}.pkl"))
            return models, "live"
        except Exception:
            pass  # fall through to cached version

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
    st.markdown(
        """<div style="background:#fff7ed; border:1px solid #fed7aa; border-left:4px solid #ea580c;
        border-radius:0.6rem; padding:0.6rem 1rem; margin-bottom:1rem;">
        <span style="color:#9a3412; font-weight:600;">⚠️ Showing cached data — live connection to Hopsworks is temporarily unavailable.</span>
        </div>""",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """<div style="background:#f0fdf4; border:1px solid #bbf7d0; border-left:4px solid #16a34a;
        border-radius:0.6rem; padding:0.6rem 1rem; margin-bottom:1rem;">
        <span style="color:#166534; font-weight:600;">Connected to live Hopsworks Feature Store</span>
        </div>""",
        unsafe_allow_html=True
    )

latest = df.iloc[-1]
category, color = aqi_category(latest["aqi"])

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "3-Day Forecast", "Historical Trends", "Model Insights"])

# ============ TAB 1: OVERVIEW ============
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current AQI", int(latest["aqi"]))
    col2.metric("PM2.5", f"{latest['pm2_5']:.1f}")
    col3.metric("Temperature", f"{latest['temperature']:.1f}°C")
    col4.metric("Humidity", f"{latest['humidity']:.0f}%")

    st.markdown(
        f"""<div style="padding: 1.1rem 1.3rem; border-radius: 0.9rem; background: linear-gradient(135deg, {color}18, {color}08);
        border-left: 5px solid {color}; margin-top: 0.8rem; box-shadow: 0 2px 8px {color}15;">
        <h3 style="color: {color} !important; margin: 0;">Air Quality: {category}</h3></div>""",
        unsafe_allow_html=True
    )
    st.write("")
    if latest["aqi"] > 150:
        st.error(f"Air quality is currently **{category}**. Consider limiting outdoor activity.")
    elif latest["aqi"] > 100:
        st.warning(f"Air quality is **{category}**. Sensitive groups should take precautions.")
    else:
        st.success(f"Air quality is **{category}**.")

    st.subheader("Current Pollutant Levels")
    pollutant_cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2"]
    st.bar_chart(latest[pollutant_cols], color="#0ea5e9")
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
                f"""<div style="padding: 1.4rem; border-radius: 1rem; background: linear-gradient(160deg, {pred_color}14, {pred_color}05);
                border: 1.5px solid {pred_color}55; text-align: center; box-shadow: 0 4px 14px {pred_color}18;">
                <h1 style="color: {pred_color} !important; margin: 0; font-size: 2.6rem; font-family:'Poppins',sans-serif;">{int(pred)}</h1>
                <p style="color: {pred_color} !important; margin: 0.3rem 0 0 0; font-weight: 700;">{pred_cat}</p>
                </div>""",
                unsafe_allow_html=True
            )

    st.info("Forecasts generated using Random Forest models trained separately for each time horizon (24h, 48h, 72h ahead), using historical pollutant and weather patterns.")

# ============ TAB 3: HISTORICAL TRENDS ============
with tab3:
    st.subheader("AQI Trend (Last 7 Days)")
    st.line_chart(df.set_index("timestamp")["aqi"].tail(168), color="#14b8a6")

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