# TF-DF Funding Predictor API

A Flask API that mirrors your Streamlit preprocessing and serves predictions to a Next.js frontend.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Ensure models_new_city/ and artifacts/ are present in this directory
python wsgi.py