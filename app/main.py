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
from app.ML.models import load_model
import pickle
from pathlib import Path


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

MODEL_PATH = Path("ML") / "models" / "nurse_shift_model.pkl"

def load_model(file_path: Path):
    """Loads a pickle file (ML model)."""
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from: {file_path}")
        return model
    except FileNotFoundError:
        print(f"ERROR: Model not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        return None

nurse_shift_model = load_model(MODEL_PATH)

if nurse_shift_model:
    pass


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
GET /score-shifts/{nurse_id} API endpoint

This function returns scored shift recommendations for a given nurse.

NOTE: This is a prototype. The function uses SYNTHETIC/FALLBACK data 
for fields that may be missing in the database. These synthetic values are 
meant as placeholders and should be replaced with real data when available.

SYNTHETIC fields are marked with "# SYNTHETIC" in the code.
"""


    def safe_float(value, default):
        try:
            if value is None or value == "":
                return float(default)
            return float(value)
        except:
            return float(default)

    def safe_distance(row):
        if pd.isna(row['lat']) or pd.isna(row['lng']) or pd.isna(row['base_lat']) or pd.isna(row['base_lng']):
            return 0.0
        return geodesic((row['base_lat'], row['base_lng']), (row['lat'], row['lng'])).km

    
    nurse = db.query(Nurse).filter(Nurse.id == nurse_id).first()
    if not nurse:
        raise HTTPException(status_code=404, detail="Nurse not found")

    shifts = db.query(ShiftAdvertisement).all()
    if not shifts:
        return {"message": "No shifts available"}

    
    shifts_df = pd.DataFrame([{
        "id_shift": str(s.id),
        "specialization_shift": getattr(s, "specialization", None),
        "location": getattr(s, "location", None),
        "shift_start_date": pd.to_datetime(getattr(s, "shift_start_date", pd.NaT), errors='coerce'),
        "shift_end_date": pd.to_datetime(getattr(s, "shift_end_date", pd.NaT), errors='coerce'),
        "shift_type": getattr(s, "shift_type", None),
        "lat": safe_float(getattr(s, "lat", None), np.random.uniform(55.3, 69.1)),# SYNTHETIC
        "lng": safe_float(getattr(s, "lng", None), np.random.uniform(11.1, 24.2)),# SYNTHETIC
        "hospital_id": int(getattr(s, "hospital_id", -1)) if getattr(s, "hospital_id", None) not in [None, ""] else -1,
        "role_id_shift": int(getattr(s, "role_id", 0)) if getattr(s, "role_id", None) not in [None, ""] else 0,
        "experience_required": pd.to_numeric(getattr(s, "experience_required", 0), errors='coerce') or 0
    } for s in shifts])

    
    nurse_df = pd.DataFrame([{
        "id_nurse": str(nurse.id),
        "specialization_nurse": getattr(nurse, "specialization", None),
        "role_id_nurse": int(getattr(nurse, "role_id", 0)) if getattr(nurse, "role_id", None) not in [None, ""] else 0,
        "base_lat": safe_float(getattr(nurse, "base_lat", None), np.random.uniform(55.3, 69.1)),# SYNTHETIC 
        "base_lng": safe_float(getattr(nurse, "base_lng", None), np.random.uniform(11.1, 24.2)),# SYNTHETIC 
        "age": ((pd.Timestamp('today') - pd.to_datetime(getattr(nurse, "date_of_birth", pd.Timestamp('2000-01-01')), errors='coerce')).days // 365) or 30,#SYNTHETIC FALLBACK
        "avg_distance_accepted": safe_float(getattr(nurse, "avg_distance_accepted", None), np.random.uniform(0, 200)),# SYNTHETIC placeholder
        "night_shift_preference": safe_float(getattr(nurse, "night_shift_preference", None), np.random.uniform(0, 1)),# SYNTHETIC placeholder
        "avg_hourly_rate_accepted": safe_float(getattr(nurse, "avg_hourly_rate_accepted", None), np.random.uniform(200, 600)),# SYNTHETIC placeholder
        "preferred_locations": getattr(nurse, "preferred_locations", []) or [],
        "total_shifts_completed": pd.to_numeric(getattr(nurse, "total_shifts_completed", 0), errors='coerce') or 0
    }])


    pairs_df = nurse_df.assign(key=1).merge(shifts_df.assign(key=1), on='key').drop('key', axis=1)

    
    pairs_df['distance_km'] = pairs_df.apply(safe_distance, axis=1)
    pairs_df['specialization_match'] = (pairs_df['specialization_nurse'] == pairs_df['specialization_shift']).astype(int)
    pairs_df['role_match'] = (pairs_df['role_id_nurse'] == pairs_df['role_id_shift']).astype(int)
    pairs_df['location_preference_match'] = pairs_df.apply(lambda r: int(r['location'] in r['preferred_locations']), axis=1)
    pairs_df['shift_start_date'] = pd.to_datetime(pairs_df['shift_start_date'])
    pairs_df['shift_end_date'] = pd.to_datetime(pairs_df['shift_end_date'], errors='coerce')
    pairs_df['shift_hour'] = pairs_df['shift_start_date'].dt.hour.fillna(0)
    pairs_df['shift_day_of_week'] = pairs_df['shift_start_date'].dt.dayofweek.fillna(0)
    pairs_df['shift_duration_hours'] = ((pairs_df['shift_end_date'] - pairs_df['shift_start_date']).dt.total_seconds() / 3600).fillna(8)
    pairs_df['is_night_shift'] = ((pairs_df['shift_hour'] >= 20) | (pairs_df['shift_hour'] < 6)).astype(int)
    pairs_df['lead_time_days'] = (pairs_df['shift_start_date'] - pd.Timestamp('today')).dt.days.fillna(1)  # SYNTHETIC fallback
    pairs_df['experience_gap'] = pd.to_numeric(pairs_df['total_shifts_completed'], errors='coerce').fillna(0) - pd.to_numeric(pairs_df['experience_required'], errors='coerce').fillna(0)# SYNTHETIC fallback
    pairs_df['preferred_location_distance_rank'] = 1

    
    hospital_ids = shifts_df['hospital_id'].dropna().unique().tolist()
    for hid in hospital_ids:
        col_name = f"hospital_{hid}_familiarity"
        if col_name not in pairs_df.columns:
            pairs_df[col_name] = 0 # SYNTHETIC placeholder

    scored_df = score_shifts(pairs_df, hospital_ids)

    scored_df['id_nurse'] = pairs_df['id_nurse']
    scored_df['id_shift'] = pairs_df['id_shift']

    recommended = scored_df.sort_values(by="pred_score", ascending=False).replace({np.nan: None, np.inf: None, -np.inf: None})
    return recommended.to_dict(orient="records")


