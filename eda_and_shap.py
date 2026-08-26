import os
import hopsworks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from dotenv import load_dotenv

load_dotenv()

# Create output folder for saved plots
os.makedirs("eda_outputs", exist_ok=True)

# ---- Step 1: Load data and model (ONE-TIME connection to save quota) ----
print("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
mr = project.get_model_registry()

print("Loading data...")
fg = fs.get_feature_group("aqi_features", version=3)
df = fg.read(read_options={"use_hive": True})
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Loaded {len(df)} rows")

print("Loading model...")
model_obj = mr.get_model("aqi_predictor", version=1)
model_dir = model_obj.download()
model = joblib.load(os.path.join(model_dir, "rf_model.pkl"))

feature_cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2",
                 "hour", "day_of_week", "month",
                 "temperature", "humidity", "wind_speed"]

# ============================================
# EDA (Exploratory Data Analysis)
# ============================================

# 1. AQI trend over time
plt.figure(figsize=(14, 5))
plt.plot(df["timestamp"], df["aqi"], linewidth=0.7)
plt.title("AQI Trend Over Time (Lahore)")
plt.xlabel("Date")
plt.ylabel("AQI (US Scale)")
plt.tight_layout()
plt.savefig("eda_outputs/1_aqi_trend.png")
plt.close()
print("Saved: AQI trend chart")

# 2. Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df[feature_cols + ["aqi"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_outputs/2_correlation_heatmap.png")
plt.close()
print("Saved: Correlation heatmap")

# 3. AQI distribution by hour of day
plt.figure(figsize=(12, 5))
sns.boxplot(x="hour", y="aqi", data=df)
plt.title("AQI Distribution by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig("eda_outputs/3_aqi_by_hour.png")
plt.close()
print("Saved: AQI by hour")

# 4. AQI distribution by month (seasonal pattern)
plt.figure(figsize=(12, 5))
sns.boxplot(x="month", y="aqi", data=df)
plt.title("AQI Distribution by Month (Seasonal Pattern)")
plt.xlabel("Month")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig("eda_outputs/4_aqi_by_month.png")
plt.close()
print("Saved: AQI by month")

# 5. PM2.5 vs AQI scatter
plt.figure(figsize=(8, 6))
plt.scatter(df["pm2_5"], df["aqi"], alpha=0.3, s=5)
plt.title("PM2.5 vs AQI")
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig("eda_outputs/5_pm25_vs_aqi.png")
plt.close()
print("Saved: PM2.5 vs AQI scatter")

# ============================================
# SHAP (Model Explainability)
# ============================================
print("\nComputing SHAP values (this may take a minute)...")

# Use a sample of data for speed (SHAP can be slow on large datasets)
X_sample = df[feature_cols].sample(n=min(500, len(df)), random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Summary plot (shows feature importance + impact direction)
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("eda_outputs/6_shap_summary.png")
plt.close()
print("Saved: SHAP summary plot")

# Bar plot (simpler version, just importance ranking)
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("eda_outputs/7_shap_importance_bar.png")
plt.close()
print("Saved: SHAP importance bar chart")

print("\n✅ All done! Check the 'eda_outputs' folder for images.")