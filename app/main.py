from fastapi import FastAPI,Depends,HTTPException
from uuid import UUID
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.nurse import Nurse
from app.models.shift_advertisement import ShiftAdvertisement
from app.models.associations import nurse_shift
from app.schemas import ShiftSchema
from app.schemas import NurseWithShiftsSchema
from app.schemas import NurseBaseSchema
from app.ML.inference.score_shifts import score_shifts
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import inspect
from typing import Any,List,Dict
from app.ML.models import load_model

# =========================
# SYNTHETIC / PLACEHOLDER FIELDS
# Replace these with real data when available
# - base_lat, base_lng
# - avg_distance_accepted, night_shift_preference, avg_hourly_rate_accepted
# - lead_time_days, experience_gap
# - role_match (currently always 1)
# - expected_features feature_11–feature_33 placeholders
# Fallbacks for shifts:
# - lat, lng if missing
# =========================



app = FastAPI()

def _normalize_none(val):
    return None if val is None else val

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/dev-data/{nurse_id}",response_model= NurseWithShiftsSchema)
def get_dev_data(nurse_id: str, db: Session = Depends(get_db)):
    nurse = db.query(Nurse).filter(Nurse.id == nurse_id).first()
    if not nurse:
        raise HTTPException(status_code=404, detail="Nurse not found")
    return NurseWithShiftsSchema.from_orm(nurse)

@app.post("/assign-shift/{nurse_id}/{shift_id}",response_model = NurseWithShiftsSchema)
def assign_shift_to_nurse(nurse_id: UUID, shift_id: UUID, db: Session = Depends(get_db)):
    nurse = db.query(Nurse).filter(Nurse.id == nurse_id).first()
    if not nurse:
        raise HTTPException(status_code=404, detail="Nurse not found")
    shift = db.query(ShiftAdvertisement).filter(ShiftAdvertisement.id == shift_id).first()
    if not shift:
        raise HTTPException(status_code=404, detail="Shift not found")

    exists = db.query(nurse_shift).filter(
        nurse_shift.c.nurse_id == nurse_id,
        nurse_shift.c.shift_id == shift_id
    ).first()
    if not exists:
        db.execute(nurse_shift.insert().values(nurse_id=str(nurse_id), shift_id=str(shift_id)))
        db.commit()
        db.refresh(nurse)

    return NurseWithShiftsSchema.from_orm(nurse)


@app.get("/all-shifts",response_model=list[ShiftSchema])
def get_all_shifts(db: Session = Depends(get_db)):
    shifts = db.query(ShiftAdvertisement).all()
    return [ShiftSchema.from_orm(s) for s in shifts]

@app.get("/nurses",response_model=list[NurseBaseSchema])
def get_all_nurses(db: Session = Depends(get_db)):
    nurses = db.query(Nurse).all()
    return [NurseBaseSchema.from_orm(s) for s in nurses]

@app.get("/score-shifts/{nurse_id}")
def get_shift_scores(nurse_id: UUID, db: Session = Depends(get_db)):
    """
    Score all available shifts for a given nurse_id and return
    match_score per shift sorted descending.

    Prepares nurse-shift pairs with correct numeric features for
    the trained LightGBM model.

    NOTE: This implementation uses synthetic data for missing fields.
    Replace synthetic data generation with real historical data
    """
    nurse = db.query(Nurse).filter(Nurse.id == nurse_id).first()
    if not nurse:
        raise HTTPException(status_code=404, detail="Nurse not found")

    shifts = db.query(ShiftAdvertisement).all()
    if not shifts:
        return {"message": "No shifts available"}

    shifts_df = pd.DataFrame([{
        "id_shift": s.id,
        "specialization_shift": _normalize_none(getattr(s, "specialization", None)),
        "location": _normalize_none(getattr(s, "location", None)),
        "shift_start_date": getattr(s, "shift_start_date", pd.NaT),
        "shift_type": _normalize_none(getattr(s, "shift_type", None)),
        "lat": float(getattr(s, "lat")) if getattr(s, "lat", None) is not None else np.nan,
        "lng": float(getattr(s, "lng")) if getattr(s, "lng", None) is not None else np.nan,
        "hospital_id": int(getattr(s, "hospital_id", -1)) if getattr(s, "hospital_id", None) is not None else -1,
    } for s in shifts])

    if shifts_df['shift_start_date'].isnull().all():
        raise HTTPException(status_code=500, detail="No valid shifts found (missing start dates)")

    nurse_df = pd.DataFrame([{
        "id_nurse": str(nurse.id),
        "specialization_nurse": _normalize_none(getattr(nurse, "specialization", None)),
        "role_id": int(getattr(nurse, "role_id", 0)),
        "base_lat": float(getattr(nurse, "base_lat", np.random.uniform(55.3, 69.1))),
        "base_lng": float(getattr(nurse, "base_lng", np.random.uniform(11.1, 24.2))),
        "age": ((pd.Timestamp('today') - pd.to_datetime(getattr(nurse, "date_of_birth", pd.Timestamp('2000-01-01')))).days // 365),
        "avg_distance_accepted": float(getattr(nurse, "avg_distance_accepted", np.random.uniform(0, 200))),
        "night_shift_preference": float(getattr(nurse, "night_shift_preference", np.random.uniform(0,1))),
        "avg_hourly_rate_accepted": float(getattr(nurse, "avg_hourly_rate_accepted", np.random.uniform(200,600))),
        "preferred_locations": getattr(nurse, "preferred_locations", []) or []
    }])

    pairs_df = nurse_df.assign(key=1).merge(shifts_df.assign(key=1), on='key').drop('key', axis=1)

    def safe_distance(row):
        if pd.isna(row['lat']) or pd.isna(row['lng']) or pd.isna(row['base_lat']) or pd.isna(row['base_lng']):
            return np.nan
        return geodesic((row['base_lat'], row['base_lng']), (row['lat'], row['lng'])).km


    pairs_df['distance_km'] = pairs_df.apply(safe_distance, axis=1) # SYNTHETIC if base_lat/lng or shift lat/lng missing
    pairs_df['specialization_match'] = (pairs_df['specialization_nurse'] == pairs_df['specialization_shift']).astype(int)
    pairs_df['role_match'] = 1  # SYNTHETIC placeholder, replace with real role matching logic
    pairs_df['location_preference_match'] = pairs_df.apply(
        lambda r: int(r['location'] in r['preferred_locations']), axis=1
    )
    pairs_df['is_night_shift'] = ((pairs_df['shift_start_date'].dt.hour >= 20) |
                                (pairs_df['shift_start_date'].dt.hour < 6)).astype(int)
    pairs_df['lead_time_days'] = np.random.randint(1, 30, len(pairs_df))  # SYNTHETIC
    pairs_df['experience_gap'] = np.random.randint(-5, 5, len(pairs_df))   # SYNTHETIC

    for col in ['specialization_nurse', 'role_id', 'shift_type', 'location']:
        if col in pairs_df.columns:
            pairs_df[col] = pairs_df[col].astype('category').cat.codes

    expected_features = [
        'age', 'distance_km', 'specialization_match', 'role_match',
        'experience_gap', 'is_night_shift', 'lead_time_days',
        'location_preference_match', 'avg_distance_accepted',
        'night_shift_preference', 'avg_hourly_rate_accepted'
    ] + [f'feature_{i}' for i in range(11, 34)] # SYNTHETIC placeholders for future features

    X = pd.DataFrame(0, index=pairs_df.index, columns=expected_features)
    for col in ['age', 'distance_km', 'specialization_match', 'role_match',
                'experience_gap', 'is_night_shift', 'lead_time_days',
                'location_preference_match', 'avg_distance_accepted',
                'night_shift_preference', 'avg_hourly_rate_accepted']:
        X[col] = pairs_df[col]

    try:
        scored_df = score_shifts(X) # SYNTHETIC target predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring shifts: {e}")

    scored_df['id_nurse'] = pairs_df['id_nurse']
    scored_df['id_shift'] = pairs_df['id_shift']

    recommended = scored_df.sort_values(by="pred_score", ascending=False)

    recommended = recommended.replace({np.nan: None, np.inf: None, -np.inf:None})
    return recommended.to_dict(orient="records")


