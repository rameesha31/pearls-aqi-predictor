import requests
import pandas as pd
import hopsworks
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

LAT = 31.5497
LON = 74.3436

# Fixed date range
START_DATE = datetime(2025, 7, 25).date()
END_DATE = datetime(2026, 8, 24).date()

def fetch_historical_weather():
    """Fetch hourly weather data from Open-Meteo (free, no API key needed)"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    return response.json()

def fetch_historical_air_quality():
    """Fetch hourly pollutant + AQI data from Open-Meteo Air Quality API"""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide,sulphur_dioxide,us_aqi",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    return response.json()

def build_dataframe(weather_json, air_json):
    """Combine weather + air quality into one clean DataFrame"""
    weather_df = pd.DataFrame({
        "timestamp": pd.to_datetime(weather_json["hourly"]["time"]),
        "temperature": weather_json["hourly"]["temperature_2m"],
        "humidity": weather_json["hourly"]["relative_humidity_2m"],
        "wind_speed": weather_json["hourly"]["wind_speed_10m"],
    })

    air_df = pd.DataFrame({
        "timestamp": pd.to_datetime(air_json["hourly"]["time"]),
        "pm2_5": air_json["hourly"]["pm2_5"],
        "pm10": air_json["hourly"]["pm10"],
        "no2": air_json["hourly"]["nitrogen_dioxide"],
        "o3": air_json["hourly"]["ozone"],
        "co": air_json["hourly"]["carbon_monoxide"],
        "so2": air_json["hourly"]["sulphur_dioxide"],
        "aqi": air_json["hourly"]["us_aqi"],
    })

    df = pd.merge(weather_df, air_df, on="timestamp")
    df = df.dropna()

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    float_cols = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10",
                  "no2", "o3", "co", "so2"]
    int_cols = ["hour", "day_of_week", "month", "aqi"]

    for col in float_cols:
        df[col] = df[col].astype(float)
    for col in int_cols:
        df[col] = df[col].astype(int)

    return df

def push_to_hopsworks(df):
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=3,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Hourly AQI features from Open-Meteo (weather + air quality)",
        time_travel_format="HUDI"
    )
    fg.insert(df)
    print("Data pushed to Hopsworks (version 2)!")

if __name__ == "__main__":
    print(f"Fetching data from {START_DATE} to {END_DATE}...")

    print("Fetching weather...")
    weather_json = fetch_historical_weather()

    print("Fetching air quality...")
    air_json = fetch_historical_air_quality()

    print("Building dataframe...")
    df = build_dataframe(weather_json, air_json)
    print(df.head())
    print("Total rows:", len(df))

    print("Pushing to Hopsworks...")
    push_to_hopsworks(df)