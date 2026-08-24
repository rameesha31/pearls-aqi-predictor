import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

print("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=1)

print("Reading data (using Hive fallback)...")
df = fg.read(read_options={"use_hive": True})

print("Data read complete!")
print(df.tail(10))
print("\nTotal rows:", len(df))