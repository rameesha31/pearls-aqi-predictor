import os
import hopsworks
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

load_dotenv()

# ---- Step 1: Data Fetch Karo Hopsworks Se ----
def load_data():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=3)
    df = fg.read(read_options={"use_hive": True})
    return df

# ---- Step 2: Data Ko Prepare Karo ----
# def prepare_data(df):
#     # Time ke hisaab se sort karo (zaroori hai time-series ke liye)
#     df = df.sort_values("timestamp").reset_index(drop=True)

#     # Features (input) aur Target (output) alag karo
#     feature_cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2", "nh3",
#                      "hour", "day_of_week", "month",
#                      "temperature", "humidity", "wind_speed"]
#     target_col = "aqi"

#     X = df[feature_cols]
#     y = df[target_col]

#     # Time ke hisaab se split (last 20% test, baaki training)
#     split_index = int(len(df) * 0.8)
#     X_train, X_test = X[:split_index], X[split_index:]
#     y_train, y_test = y[:split_index], y[split_index:]

#     print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")
#     return X_train, X_test, y_train, y_test, feature_cols
def prepare_data(df, horizon_hours=24):
    # Time ke hisaab se sort karo
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2",
                     "hour", "day_of_week", "month",
                     "temperature", "humidity", "wind_speed"]

    # Target ko shift karo — future ka AQI is row ka target banega
    df["future_aqi"] = df["aqi"].shift(-horizon_hours)

    # Jin rows ka future data nahi hai (aakhri N rows), unhe hata do
    df = df.dropna(subset=["future_aqi"])

    X = df[feature_cols]
    y = df["future_aqi"]

    split_index = int(len(df) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Forecasting {horizon_hours} hours ahead")
    print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_cols
# ---- Step 3: Model Train Karo ----
def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}

    # Model 1: Ridge Regression (baseline, simple)
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    results["Ridge"] = {
        "model": ridge,
        "rmse": np.sqrt(mean_squared_error(y_test, ridge_preds)),
        "mae": mean_absolute_error(y_test, ridge_preds),
        "r2": r2_score(y_test, ridge_preds),
    }

    # Model 2: Random Forest (zyada powerful)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results["RandomForest"] = {
        "model": rf,
        "rmse": np.sqrt(mean_squared_error(y_test, rf_preds)),
        "mae": mean_absolute_error(y_test, rf_preds),
        "r2": r2_score(y_test, rf_preds),
    }

    # Results print karo
    print("\n--- Model Comparison ---")
    for name, res in results.items():
        print(f"{name}: RMSE={res['rmse']:.3f}, MAE={res['mae']:.3f}, R2={res['r2']:.3f}")

    return results
# ---- Step 4: Save Best Model to Registry ----
def save_model_to_registry(rf_model, X_train):
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    mr = project.get_model_registry()

    # Save model file locally first
    os.makedirs("model_dir", exist_ok=True)
    joblib.dump(rf_model, "model_dir/rf_model.pkl")

    # Register in Hopsworks Model Registry
    model = mr.python.create_model(
        name="aqi_predictor",
        description="Random Forest model for 24-hour AQI forecasting"
    )
    model.save("model_dir")
    print("Model saved to registry!")

# ---- Main ----
if __name__ == "__main__":
    print("Loading data from Hopsworks...")
    df = load_data()
    print(f"Loaded {len(df)} rows")

    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df, horizon_hours=24)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Save the Random Forest model (better performer)
    save_model_to_registry(results["RandomForest"]["model"], X_train)