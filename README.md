# aSeedic Server

Flask-based REST API for startup intelligence — funding prediction, scoring, investor matching, location recommendations, and AI-powered pitch deck generation.

---

## Features

- **Funding Prediction** — TF-Decision Forests model predicts total fundable amount
- **QFS Scoring** — QuantumFAI Score (0-100) with token allocation and valuation estimate
- **Investor Matching** — TF-IDF cosine similarity matching against investor database
- **Location Recommendation** — Composite scoring across density, funding, success rate, and growth
- **Pitch Deck Generation** — Gemini-powered 10-slide pitch deck with PPTX export
- **LLM Recommendations** — Gemini-generated actionable insights on top of each model's output

---

## Requirements

- Python 3.11
- Google Gemini API key

---

## Running Locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install tensorflow==2.15.0 tf-keras==2.15.1 tensorflow-decision-forests==1.8.1
python wsgi.py
```

Server starts at `http://localhost:8000`

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key |
| `GEMINI_MODEL` | No | `gemini-2.0-flash` | Gemini model to use |
| `MODEL_TYPE` | No | `combined` | ML model variant: `combined`, `city`, `state`, `money` |
| `CORS_ALLOW_ORIGINS` | No | `*` | Comma-separated allowed origins |

---

## API Endpoints

### Health
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |

### Prediction
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/funding/predict` | Predict total funding amount |
| POST | `/api/score` | QFS score, tokens, and valuation for a batch of startups |
| POST | `/api/match` | Match startup with investors |

### Location
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/location/prep` | Prepare location data (run once) |
| POST | `/api/location/train` | Train location recommender model |
| POST | `/api/location/recommend` | Get top locations for an industry |
| GET | `/api/location/industries` | List available industries |

### Pitch Deck
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/pitch/generate-full-deck` | Generate all 10 slides (JSON) |
| POST | `/api/pitch/generate-slide` | Regenerate a single slide |
| POST | `/api/pitch/generate-ppt` | Export slides to PPTX |
| POST | `/api/pitch/generate-and-download` | Generate + export in one call |

### LLM Recommendations
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/recommend/funding` | Funding prediction + Gemini recommendation |
| POST | `/api/recommend/score` | QFS score + Gemini recommendation |
| POST | `/api/recommend/investors` | Investor matches + outreach strategy |
| POST | `/api/recommend/location` | Location rankings + expansion strategy |

---

## Project Structure

```
app/
  routes/         — Blueprint route handlers (controllers)
  services/       — Business logic layer
  schemas.py      — Pydantic request/response models
  config.py       — Configuration and model path resolution
models_*/         — Saved TF-Decision Forests models (4 variants)
artifacts_*/      — Model artifacts: feature order, class encodings, metadata
investor_db.csv   — Investor database for matching
filled_tf_df.csv  — Training data for location recommender
wsgi.py           — Entry point
```
