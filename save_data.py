import os
import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=3)
df = fg.read(read_options={"use_hive": True})
df = df.sort_values("timestamp").reset_index(drop=True)

df.to_csv("data_snapshot.csv", index=False)
print(f"Saved {len(df)} rows to data_snapshot.csv")