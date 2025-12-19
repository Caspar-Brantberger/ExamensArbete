# PH Suggestion Engine

## Overview
This project is a machine learning-powered recommendation engine for nurse shift assignments. It uses nurse and shift data to generate ranked shift recommendations using a LightGBM classifier. FastAPI serves as the backend for both production and internal/dev endpoints.

## Table of Contents
1. Features
2. Dataset
3. Training the Model
4. Retraining the Model
5. Deployment
6. FastAPI Endpoints
7. Mock Token Generation
8. Dependencies
9. Versioning Strategy
10. Notes on Placeholder/Synthetic Data
11. Unit Tests

## Features
The ML model uses the following features:
- age – Nurse's age (calculated from date_of_birth).
- distance_km – Distance from nurse base to shift location (geodesic).
- specialization_match – 1 if nurse specialization matches shift, else 0.
- role_match – 1 if nurse role matches shift role, else 0.
- experience_gap – Difference between nurse experience (total_shifts_completed) and required experience for shift.
- is_night_shift – 1 if shift starts between 20:00–06:00, else 0.
- lead_time_days – Days between now and shift start.
- location_preference_match – 1 if shift location is among nurse preferred locations, else 0.
- avg_distance_accepted, night_shift_preference, avg_hourly_rate_accepted – Placeholder synthetic features (replace with real data).
- shift_hour, shift_day_of_week, shift_duration_hours – Derived from shift start/end times.
- preferred_location_distance_rank – Placeholder ranking.
- hospital_{hospital_id}_familiarity – One column per hospital indicating familiarity.

> All synthetic/placeholder data are marked in `build_pairs_with_features.py` and `train_model.py`. Replace them with real data when available.

Type hints for `build_pairs_with_features`:
```python
def build_pairs_with_features(
    nurses: pd.DataFrame,
    shifts: pd.DataFrame,
    nurse_shifts: pd.DataFrame
) -> tuple[pd.DataFrame, list]:
    ...

Dataset

Tables used:

    nurse – Contains nurse info including specialization, base location, completed shifts.

    shift_advertisement – Contains shift details including hospital, department, specialization required, start/end time, location.

    nurse_shift – Historical nurse-to-shift assignments used for labeling.

Database credentials in .env:

DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=
DB_NAME=
ISSUER_SECRET=dG9wU2VjMTIz

Training the Model

    Activate virtual environment:

.\venv\Scripts\Activate   # Windows
source venv/bin/activate  # Mac/Linux

    Run training script:

python -m app.ML.training.train_model

The script loads data from the database, builds pairwise features, trains a LightGBM classifier, and saves the model as models/nurse_shift_model_<YYYY-MM-DD>.pkl.

Check train_model.py for FEATURE_COLS_BASE and update when adding new features.
Retraining the Model

    Ensure .env has correct DB credentials.

    Activate venv.

    Run:

python -m app.ML.training.train_model

    Confirm new model is saved in models/.

    Restart FastAPI backend to use the new model.

    Check build_pairs_with_features.py for columns to include in features.

Deployment

    Clone repo.

    Install dependencies:

pip install -r requirements.txt

    Configure .env.

    Start FastAPI:

uvicorn app.main:app --reload

Server available at http://127.0.0.1:8000.
FastAPI Endpoints

Public

    GET /health – Health check.

    POST /v1/recommendations/shifts – Returns ranked shift recommendations for a nurse. Accepts either:

{ "nurse_id": "...", "max_results": 20 }

or

{
  "nurse": { "id": "..." },
  "candidate_shifts": [{ "id": "..." }]
}

Internal / Dev

    GET /internal/dev-data/{nurse_id} – Returns nurse data with shifts.

    GET /internal/nurses – Returns all nurses.

    GET /internal/shifts – Returns all shifts.

    GET /internal/recommend/{nurse_id} – Returns placeholder recommendations.

Mock Token Generation

Generate a mock JWT for testing:

python -m app.scripts.generate_mock_token

Use in headers:

Authorization: Bearer <token>

Dependencies

Key dependencies:

fastapi
uvicorn
sqlalchemy
pandas
numpy
lightgbm
geopy
python-dotenv
pyjwt
pydantic
joblib

Install all with:

pip install -r requirements.txt

Versioning Strategy

    Models saved with date: nurse_shift_model_<YYYY-MM-DD>.pkl.

    Track modifications in changelog.

    Only latest stable model should be used in production.

Notes on Placeholder / Synthetic Data

    All synthetic data marked in build_pairs_with_features.py and train_model.py with # placeholder or # SYNTHETIC.

    Replace placeholders with real data: base_lat, base_lng, avg_distance_accepted, night_shift_preference, avg_hourly_rate_accepted, preferred_locations, experience_required, role_id, distance ranking.

Unit Tests

Run in venv:

pytest tests/test_transform.py

All tests should pass:

collected 4 items
tests\test_transform.py .... [100%]

Summary

This README covers:

    Feature definitions

    Training process

    Retraining steps

    Deployment instructions

    Versioning strategy

    Dataset and DB setup

    FastAPI endpoints

    Mock token instructions

    Dependencies

    Unit tests

    Notes on synthetic data