# PH Suggestion Engine
This project is a nurse shift recommendation feature that I developed during my internship. The company decided to discontinue the project and gave me permission to continue working on it independently, as I wrote all the code myself.
"This is a nurse shift recommendation engine I built. It uses a machine learning model to score and rank shifts based on each nurse’s profile, like experience, specialization, and location. The top-ranked shifts are the ones that best match the nurse, and all requests are securely authenticated using JWT tokens."


## Local Database Setup for Testing

If you want to test the project with a local copy of the database, follow these steps:

1. **Download the database file**  
   - The SQL dump `dump-phdb-202512291404.sql` is available in the `db` folder. Save it locally.

2. **Create a new database in PostgreSQL**  
   - Use DBeaver or `psql` to create a database, e.g., `ph_suggestion_engine_test`.

3. **Import the SQL file**  
   - **Via DBeaver:**  
     1. Right-click on the new database → **Tools → Restore** or **Execute Script**.  
     2. Select the `dump-phdb-202512291404.sql` file.  
     3. Run the import.  
   - **Via terminal:**  
     ```bash
     psql -U <db_user> -d ph_suggestion_engine_test -f path/to/ph_suggestion_engine.sql
     ```
     Replace `<db_user>` with your PostgreSQL username and `path/to/` with the SQL file path.

4. **Update `.env`**  
   - Create a `.env` file in the backend with the database credentials:  
     ```env
     DB_USER=<your_user>
     DB_PASSWORD=<your_password>
     DB_HOST=localhost
     DB_PORT=5432
     DB_NAME=ph_suggestion_engine_test
     ```

5. **Start the backend server**  
   ```bash
   uvicorn app.main:app --reload


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

    Frontend Setup

The frontend is a React + TypeScript application.

Go to the ui folder:

cd ui


Install dependencies:

npm install


Start the development server:

npm run dev


Open http://localhost:5173
 in your browser.

Important: Replace <your_api_token_here> in the frontend code with a valid JWT token to fetch recommendations. You can generate it using the backend script:

python -m app.scripts.generate_mock_token