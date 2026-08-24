import requests

LAT = 31.5497
LON = 74.3436
START_DATE = "2025-07-25"
END_DATE = "2026-08-24"

url2 = "https://air-quality-api.open-meteo.com/v1/air-quality"
params2 = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide,sulphur_dioxide,ammonia,us_aqi",
    "timezone": "auto"
}
response2 = requests.get(url2, params=params2)
data2 = response2.json()

hourly = data2["hourly"]

# Check how many nulls in each column
for key in ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "sulphur_dioxide", "ammonia", "us_aqi"]:
    values = hourly[key]
    null_count = sum(1 for v in values if v is None)
    print(f"{key}: {null_count} nulls out of {len(values)}")

# Print a sample from the middle of the range (not the very start/end)
mid = len(hourly["time"]) // 2
print("\nSample row (middle of range):")
print("time:", hourly["time"][mid])
for key in ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "sulphur_dioxide", "ammonia", "us_aqi"]:
    print(f"{key}: {hourly[key][mid]}")