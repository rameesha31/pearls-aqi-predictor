import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

# Pehle project create karo
hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"), project=None)
project = hopsworks.create_project("a3f9c8e2_4b71_4d6a_9e0f_8c2d5b1a7f63_air_quality", description="AQI prediction project")
print("Project created:", project.name)