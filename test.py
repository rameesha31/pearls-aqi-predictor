import requests
import os
from dotenv import load_dotenv

# .env file se API key uthana
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
print("gotcha :", API_KEY)
# Lahore ka latitude/longitude (apna shehar hai to yehi rakho)
LAT = 31.5497
LON = 74.3436

# API ko request bhejna
url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
response = requests.get(url)

# Jawab print karna
print(response.status_code)
print(response.json())