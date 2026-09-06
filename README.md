# Pearls AQI Predictor

A serverless machine learning system that predicts Lahore's Air Quality Index (AQI) up to 3 days ahead. It automatically collects live weather and pollution data, engineers features, stores them in a feature store, trains models, and serves live forecasts through an interactive dashboard with no servers to manage.

**Live app:** https://pearls-aqi-predictor-lahore.streamlit.app/

---

## Setup

**1. Requirements**
- Python 3.11 (important — the `hopsworks` package doesn't install on Python 3.14+)
- A free Hopsworks account: https://app.hopsworks.ai

**2. Clone and install**
```bash
git clone https://github.com/rameesha31/pearls-aqi-predictor.git
cd pearls-aqi-predictor
py -3.11 -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

**3. Add your API key**

Create a `.env` file in the project root:
```
HOPSWORKS_API_KEY=your_key_here
```
Get this from your Hopsworks project → Account Settings → API Keys → New API Key.

**4. Windows only**

Hopsworks expects a Unix-style temp folder, which Windows doesn't have by default:
```bash
mkdir D:\tmp
```

---

## Running It

Run these once, in order, to set everything up from scratch:

```bash
python feature_pipeline.py           # creates the Feature Group, pushes current data
python backfill_historical_data.py   # pulls ~1 year of historical data (run once)
python training_pipeline.py          # trains models for 24h/48h/72h, saves to registry + locally
python eda_and_shap.py               # generates the EDA and SHAP charts used in the dashboard
```

Then run the dashboard:
```bash
streamlit run app.py
```


---

## Automation (Optional)

Two GitHub Actions workflows are already set up under `.github/workflows/`:
- `feature_pipeline.yml` — runs every hour
- `training_pipeline.yml` — runs once a day

To use them: add `HOPSWORKS_API_KEY` as a repository secret (Settings → Secrets and variables → Actions), then enable each workflow from the Actions tab.

**Note:** Hopsworks' free tier has a monthly usage quota. Running both workflows on schedule will use it up over time, so keep an eye on your usage — you can pause a workflow any time from its "..." menu without losing anything already built.

---

For the full write-up — methodology, results, and everything that went wrong (and was fixed) along the way — see `AQI_Predictor_Project_Report.docx`.
