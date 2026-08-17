import requests
import os
import hopsworks
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 31.5497
LON = 74.3436

def fetch_raw_data():
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    response = requests.get(url)
    return response.json()

def fetch_weather_data():
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    return response.json()

def build_features(pollution_data, weather_data):
    entry = pollution_data['list'][0]
    components = entry['components']
    timestamp = entry['dt']
    dt_object = datetime.fromtimestamp(timestamp)

    features = {
        "timestamp": dt_object,
        "pm2_5": float(components['pm2_5']),
        "pm10": float(components['pm10']),
        "no2": float(components['no2']),
        "o3": float(components['o3']),
        "co": float(components['co']),
        "so2": float(components['so2']),
        "nh3": float(components['nh3']),
        "hour": int(dt_object.hour),
        "day_of_week": int(dt_object.weekday()),
        "month": int(dt_object.month),
        "temperature": float(weather_data['main']['temp']),
        "humidity": int(weather_data['main']['humidity']),
        "wind_speed": float(weather_data['wind']['speed']),
        "aqi": int(entry['main']['aqi']),
    }
    return features

def push_to_feature_store(features):
    """Hopsworks se connect ho kar data push karna"""
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    # Feature Group banao (agar pehle se ho to usi se connect ho jayega)
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Hourly AQI features for Lahore",
        time_travel_format="HUDI"
    )

    # Ek row ka DataFrame banao (Hopsworks DataFrame accept karta hai)
    df = pd.DataFrame([features])

    # Insert karo
    fg.insert(df)
    print("Data pushed to Hopsworks!")

if __name__ == "__main__":
    pollution = fetch_raw_data()
    weather = fetch_weather_data()
    features = build_features(pollution, weather)
    print("Features built:", features)

    push_to_feature_store(features)