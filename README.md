# Pearls AQI Predictor

An end-to-end, serverless machine learning system that predicts the Air Quality Index (AQI) for Lahore, Pakistan, up to **3 days ahead**. It automatically collects weather and pollution data, engineers features, trains models, and serves live forecasts through an interactive dashboard.

**Live app:** https://pearls-aqi-predictor-lahore.streamlit.app/

---

## 1. Project Overview

| Stage | Tool |
|---|---|
| Data source | [Open-Meteo](https://open-meteo.com/) (Archive + Forecast + Air Quality APIs) — free, no API key needed |
| Feature Store & Model Registry | [Hopsworks Serverless](https://www.hopsworks.ai/) |
| Model training | scikit-learn (Ridge Regression, Random Forest) |
| Explainability | SHAP |
| Automation | GitHub Actions |
| Dashboard | Streamlit |

The system predicts AQI for **24h, 48h, and 72h ahead** using three separately trained Random Forest models, and displays them on a live dashboard along with historical trends, EDA charts, and model explainability plots.

---

## 2. Prerequisites

- **Python 3.11** (this project does *not* work on Python 3.14 — the `hopsworks` package fails to build on it)
- **Git**
- A **Hopsworks Serverless** free account: https://app.hopsworks.ai
- (Windows only) **Microsoft C++ Build Tools** — needed to install some Hopsworks dependencies. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/

---

## 4. Setup Instructions

### Step 1 — Clone the repository
```bash
git clone https://github.com/rameesha31/pearls-aqi-predictor.git
cd pearls-aqi-predictor
```

### Step 2 — Create a virtual environment with Python 3.11
```bash
py -3.11 -m venv venv
```
Activate it:
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```
You should see `(venv)` at the start of your terminal prompt.

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> If you install any new package later, keep `requirements.txt` up to date with:
> ```bash
> pip freeze > requirements.txt
> ```

### Step 4 — Create your `.env` file
In the project root, create a file named `.env` with:
```
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
```

To get a Hopsworks API key:
1. Sign up / log in at https://app.hopsworks.ai
2. Click **Access Hopsworks** to enter your project workspace
3. Go to **Account Settings → API Keys → New API Key**
4. Tick permissions for `featurestore`, `project`, and `model registry`
5. Copy the key immediately (it won't be shown again) and paste it into `.env`

>  `.env` is already listed in `.gitignore` — never commit it or share your key.

### Step 5 (Windows only) — Create a temp folder
The Hopsworks client expects a Unix-style `/tmp` folder, which doesn't exist by default on Windows:
```bash
mkdir D:\tmp
```
(Use whichever drive your project is on.)

---

## 5. Running the Pipelines

Run these **in order** the first time you set up the project.

### 5.1 Test the feature pipeline (fetches live data → pushes to Hopsworks)
```bash
python feature_pipeline.py
```
This creates the `aqi_features` Feature Group (version 3) in Hopsworks if it doesn't already exist, and inserts one row of the current weather/pollution reading.

### 5.2 Backfill a year of historical data (run once)
```bash
python backfill_historical_data.py
```
This pulls a full year of hourly data from Open-Meteo and pushes it to the same Feature Group. This can take a few minutes and will use a meaningful amount of your Hopsworks free-tier quota — run it only once.

### 5.3 Train the models
```bash
python training_pipeline.py
```
This reads all the data from Hopsworks, trains a Ridge Regression and a Random Forest model for each forecast horizon (24h / 48h / 72h), prints RMSE/MAE/R² for each, saves the best model (Random Forest) locally to `models/`, and pushes it to the Hopsworks Model Registry as `aqi_predictor_24h`, `aqi_predictor_48h`, and `aqi_predictor_72h`.

### 5.4 Generate EDA and SHAP charts
```bash
python eda_and_shap.py
```
This produces PNG charts in `eda_outputs/` (AQI trend, correlation heatmap, hourly/monthly patterns, SHAP importance and summary plots), which are also used inside the dashboard.

### 5.5 Create the local data snapshot (for dashboard fallback)
```bash
python -c "
import os, hopsworks, pandas as pd
from dotenv import load_dotenv
load_dotenv()
project = hopsworks.login(api_key_value=os.getenv('HOPSWORKS_API_KEY'))
fs = project.get_feature_store()
df = fs.get_feature_group('aqi_features', version=3).read(read_options={'use_hive': True})
df.sort_values('timestamp').to_csv('data_snapshot.csv', index=False)
print('Saved', len(df), 'rows')
"
```

### 5.6 Run the dashboard
```bash
streamlit run app.py
```
This opens the dashboard at `http://localhost:8501`.

> By default, `app.py` has `USE_LIVE_HOPSWORKS = False` — it reads from the local `data_snapshot.csv` and `models/` folder for speed and reliability. Set it to `True` in the code if you want it to try Hopsworks first (it will still fall back to local files automatically if the connection fails or times out).

---

## 6. Automating the Pipelines (GitHub Actions)

Two workflows are already configured under `.github/workflows/`:

| Workflow | Schedule | What it does |
|---|---|---|
| `feature_pipeline.yml` | Every hour | Runs `feature_pipeline.py` to collect the latest data |
| `training_pipeline.yml` | Every day (midnight) | Runs `training_pipeline.py` to retrain and update the models |

To enable them:
1. In your GitHub repository, go to **Settings → Secrets and variables → Actions**
2. Add a repository secret named `HOPSWORKS_API_KEY` with your Hopsworks key
3. Go to the **Actions** tab, select each workflow, and click **Enable workflow**

>  Hopsworks Serverless free tier has a monthly usage quota. If you're on the free tier, be mindful of running the feature pipeline hourly and the training pipeline daily — both consume quota. If you see your usage nearing the limit, you can disable a workflow from the "..." menu on its Actions page at any time, without losing anything you've already built.

---

## 7. Deploying the Dashboard

The app is deployed on [Streamlit Community Cloud](https://share.streamlit.io) (free):

1. Push your code (including `data_snapshot.csv`, `models/`, and `eda_outputs/`) to GitHub
2. Go to https://share.streamlit.io and sign in with GitHub
3. Click **Create app**, select your repo, branch (`main`), and main file (`app.py`)
4. Under **Advanced settings → Secrets**, add:
   ```
   HOPSWORKS_API_KEY = "your_key_here"
   ```
5. Click **Deploy**

Any time you `git push` new changes, Streamlit Cloud automatically redeploys the app within a minute or two — no need to repeat these steps.

---

## 8. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `hopsworks` fails to install / build | You're on Python 3.14+ | Use Python 3.11 in a dedicated venv (see Step 2) |
| `Microsoft Visual C++ 14.0 required` | Missing C++ Build Tools (Windows) | Install Visual Studio C++ Build Tools |
| `ModuleNotFoundError: pyarrow` / `confluent_kafka` / `delta` | Optional Hopsworks dependencies not installed | `pip install pyarrow confluent-kafka delta-spark` |
| `FileNotFoundError: /tmp/...` (Windows) | Hopsworks expects a Unix-style temp dir | Create a `D:\tmp` folder manually |
| `Features are not compatible with Feature Group schema` | Column type or name mismatch with an existing Feature Group | Cast values explicitly with `float()`/`int()`, or use a new Feature Group version if the schema has changed |
| `FlightUnavailableError` / `Socket closed` reading from Hopsworks | Temporary instability in Hopsworks' free-tier query service | Retry after a minute; the dashboard automatically falls back to `data_snapshot.csv` |
| App stuck loading forever after deployment | App is trying to log in to Hopsworks with no key/interactive terminal available (`GetPassWarning` in logs) | Set `USE_LIVE_HOPSWORKS = False` in `app.py`, or add `HOPSWORKS_API_KEY` to your deployment secrets, then reboot the app |
| `requirements.txt` missing a package after deploying | Manually edited and fell out of sync | Run `pip freeze > requirements.txt` after installing anything new, then commit and push |

---

## 9. Notes on Data & Model Choices

- **Data source:** Open-Meteo was used instead of OpenWeather for historical weather, since OpenWeather's historical weather endpoint requires a paid plan. Open-Meteo is free and provides a full year of consistent hourly data.
- **AQI scale:** The target is Open-Meteo's `us_aqi` (0–500 scale), not OpenWeather's 1–5 category scale.
- **Forecast horizons:** Three separate models are trained (24h/48h/72h) by shifting the target column forward in time, rather than predicting the AQI at the same timestamp as the input — this avoids data leakage and produces a genuine forecast.
- **Model choice:** Random Forest outperformed Ridge Regression at every horizon and is the model used in the dashboard.

For the full methodology, results, and a complete log of issues encountered and fixed during development, see the accompanying **Project Report** (`AQI_Predictor_Project_Report.docx`).
