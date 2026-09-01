import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load from local snapshot (no Hopsworks quota used)
df = pd.read_csv("data_snapshot.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

feature_cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2",
                 "hour", "day_of_week", "month",
                 "temperature", "humidity", "wind_speed"]

os.makedirs("models", exist_ok=True)

horizons = {"24h": 24, "48h": 48, "72h": 72}
all_results = {}

for label, horizon in horizons.items():
    print(f"\n--- Training models for {label} ahead ---")
    data = df.copy()
    data["future_aqi"] = data["aqi"].shift(-horizon)
    data = data.dropna(subset=["future_aqi"])

    X = data[feature_cols]
    y = data["future_aqi"]

    split_index = int(len(data) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # ---- Ridge Regression (baseline) ----
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds))
    ridge_mae = mean_absolute_error(y_test, ridge_preds)
    ridge_r2 = r2_score(y_test, ridge_preds)
    print(f"Ridge  {label}: RMSE={ridge_rmse:.3f}, MAE={ridge_mae:.3f}, R2={ridge_r2:.3f}")

    # ---- Random Forest ----
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    print(f"RF     {label}: RMSE={rf_rmse:.3f}, MAE={rf_mae:.3f}, R2={rf_r2:.3f}")

    # Save Random Forest (better performer) locally
    joblib.dump(rf, f"models/rf_{label}.pkl")
    print(f"Saved models/rf_{label}.pkl")

    all_results[label] = {
        "ridge": {"rmse": ridge_rmse, "mae": ridge_mae, "r2": ridge_r2},
        "rf": {"rmse": rf_rmse, "mae": rf_mae, "r2": rf_r2},
    }

# ---- Final summary table ----
print("\n" + "=" * 60)
print("SUMMARY — All Horizons")
print("=" * 60)
print(f"{'Horizon':<10}{'Model':<8}{'RMSE':<10}{'MAE':<10}{'R2':<10}")
for label, res in all_results.items():
    print(f"{label:<10}{'Ridge':<8}{res['ridge']['rmse']:<10.3f}{res['ridge']['mae']:<10.3f}{res['ridge']['r2']:<10.3f}")
    print(f"{label:<10}{'RF':<8}{res['rf']['rmse']:<10.3f}{res['rf']['mae']:<10.3f}{res['rf']['r2']:<10.3f}")

print("\nAll models trained and saved locally!")