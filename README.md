Pearls AQI Predictor

An end-to-end, serverless machine learning system that predicts the Air Quality Index (AQI) for Lahore, Pakistan, up to 3 days ahead. It automatically collects weather and pollution data, engineers features, trains models, and serves live forecasts through an interactive dashboard.

Live app: https://pearls-aqi-predictor-lahore.streamlit.app/

4. Setup Instructions
Step 1 — Clone the repository
bash
git clone https://github.com/rameesha31/pearls-aqi-predictor.git
cd pearls-aqi-predictor
Step 2 — Create a virtual environment with Python 3.11
bash
py -3.11 -m venv venv

Activate it:

bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

You should see (venv) at the start of your terminal prompt.

Step 3 — Install dependencies
bash
pip install -r requirements.txt

If you install any new package later, keep requirements.txt up to date with:

bash
pip freeze > requirements.txt
Step 4 — Create your .env file

In the project root, create a file named .env with:

HOPSWORKS_API_KEY=your_hopsworks_api_key_here

To get a Hopsworks API key:

Sign up / log in at https://app.hopsworks.ai
Click Access Hopsworks to enter your project workspace
Go to Account Settings → API Keys → New API Key
Tick permissions for featurestore, project, and model registry
Copy the key immediately (it won't be shown again) and paste it into .env

⚠️ .env is already listed in .gitignore — never commit it or share your key.

Step 5 (Windows only) — Create a temp folder

The Hopsworks client expects a Unix-style /tmp folder, which doesn't exist by default on Windows:

bash
mkdir D:\tmp

(Use whichever drive your project is on.)
