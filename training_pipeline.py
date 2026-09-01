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

feature_cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2",
                 "hour", "day_of_week", "month",
                 "temperature", "humidity", "wind_speed"]

# ---- Step 1: Fetch data from Hopsworks Feature Store (ONE read for all horizons) ----
def load_data():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=3)
    df = fg.read(read_options={"use_hive": True})
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, project

# ---- Step 2: Prepare data for a given forecast horizon ----
def prepare_data(df, horizon_hours):
    data = df.copy()
    data["future_aqi"] = data["aqi"].shift(-horizon_hours)
    data = data.dropna(subset=["future_aqi"])

    X = data[feature_cols]
    y = data["future_aqi"]

    split_index = int(len(data) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

# ---- Step 3: Train and evaluate both models for a horizon ----
def train_and_evaluate(X_train, X_test, y_train, y_test, label):
    print(f"\n--- Training models for {label} ahead ---")

    ridge = Ridge()
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    ridge_r2 = r2_score(y_test, ridge_preds)
    print(f"Ridge  {label}: RMSE={np.sqrt(mean_squared_error(y_test, ridge_preds)):.3f}, "
          f"MAE={mean_absolute_error(y_test, ridge_preds):.3f}, R2={ridge_r2:.3f}")

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_preds)
    print(f"RF     {label}: RMSE={np.sqrt(mean_squared_error(y_test, rf_preds)):.3f}, "
          f"MAE={mean_absolute_error(y_test, rf_preds):.3f}, R2={rf_r2:.3f}")

    return rf  # Random Forest is the selected production model

# ---- Step 4: Save model locally + push to Hopsworks Model Registry ----
def save_model(model, project, label):
    os.makedirs("models", exist_ok=True)
    local_path = f"models/rf_{label}.pkl"
    joblib.dump(model, local_path)

    model_dir = f"model_dir_{label}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, f"rf_{label}.pkl"))

    mr = project.get_model_registry()
    registry_model = mr.python.create_model(
        name=f"aqi_predictor_{label}",
        description=f"Random Forest model for {label} AQI forecasting"
    )
    registry_model.save(model_dir)
    print(f"Saved locally to {local_path} and pushed to registry as aqi_predictor_{label}")

# ---- Main ----
if __name__ == "__main__":
    print("Loading data from Hopsworks...")
    df, project = load_data()
    print(f"Loaded {len(df)} rows")

    horizons = {"24h": 24, "48h": 48, "72h": 72}

    for label, horizon in horizons.items():
        X_train, X_test, y_train, y_test = prepare_data(df, horizon)
        best_model = train_and_evaluate(X_train, X_test, y_train, y_test, label)
        save_model(best_model, project, label)

    print("\nAll models trained and registered successfully!")