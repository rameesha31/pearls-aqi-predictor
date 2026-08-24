import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read(read_options={"use_hive": True})

df = df.sort_values("timestamp").reset_index(drop=True)

print("Total rows:", len(df))
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())
print("\nDuplicate timestamps:", df["timestamp"].duplicated().sum())

print("\nTemperature stats:")
print(df["temperature"].describe())

print("\nHumidity stats:")
print(df["humidity"].describe())

print("\nAQI value counts:")
print(df["aqi"].value_counts())