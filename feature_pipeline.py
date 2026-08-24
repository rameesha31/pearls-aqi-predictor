import requests
import os
import hopsworks
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

LAT = 31.5497
LON = 74.3436

def fetch_current_weather():
    """Fetch current weather from Open-Meteo forecast API"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    return response.json()

def fetch_current_air_quality():
    """Fetch current air quality from Open-Meteo air quality API"""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide,sulphur_dioxide,us_aqi",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    return response.json()

def build_features(weather_json, air_json):
    """Combine current weather + air quality into a clean feature row"""
    w = weather_json["current"]
    a = air_json["current"]

    dt_object = pd.to_datetime(w["time"])

    features = {
        "timestamp": dt_object,
        "pm2_5": float(a["pm2_5"]),
        "pm10": float(a["pm10"]),
        "no2": float(a["nitrogen_dioxide"]),
        "o3": float(a["ozone"]),
        "co": float(a["carbon_monoxide"]),
        "so2": float(a["sulphur_dioxide"]),
        "hour": int(dt_object.hour),
        "day_of_week": int(dt_object.dayofweek),
        "month": int(dt_object.month),
        "temperature": float(w["temperature_2m"]),
        "humidity": float(w["relative_humidity_2m"]),
        "wind_speed": float(w["wind_speed_10m"]),
        "aqi": int(a["us_aqi"]),
    }
    return features

def push_to_feature_store(features):
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

    df = pd.DataFrame([features])
    fg.insert(df)
    print("Data pushed to Hopsworks (version 2)!")

if __name__ == "__main__":
    weather = fetch_current_weather()
    air = fetch_current_air_quality()
    features = build_features(weather, air)
    print("Features built:", features)

    push_to_feature_store(features)