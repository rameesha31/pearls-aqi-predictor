import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read(read_options={"use_hive": True})

df = df.sort_values("timestamp").reset_index(drop=True)

print("AQI value counts (poora dataset):")
print(df["aqi"].value_counts())

print("\nFirst 80% (training) AQI counts:")
split = int(len(df)*0.8)
print(df["aqi"][:split].value_counts())

print("\nLast 20% (testing) AQI counts:")
print(df["aqi"][split:].value_counts())