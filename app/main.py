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
from app.ML.model_loader import init as init_model
from app.ML.build_pairs_with_features import build_pairs_with_features
from app.ML.training.train_model import train_model


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

@app.on_event("startup")
def startup_event():
    init_model()



@app.get("/init-model")
async def root():
    return {"message": "Model initialized"}


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

    nurse = db.query(Nurse).filter(Nurse.id == nurse_id).first()
    if not nurse:
        raise HTTPException(status_code=404, detail="Nurse not found")
    shifts = db.query(ShiftAdvertisement).all()
    if not shifts:
        return {"message": "No shifts available"}
    
    nurse_df = pd.DataFrame([nurse.__dict__])
    shifts_df = pd.DataFrame([s.__dict__ for s in shifts])
    nurse_shifts_df = pd.DataFrame() 

    pairs_df, hospital_ids = build_pairs_with_features(nurse_df, shifts_df, nurse_shifts_df)

    pairs_df['id_nurse'] = nurse_df['id'].iloc[0]
    pairs_df['id_shift'] = shifts_df['id'].iloc[0] if len(shifts_df) == 1 else shifts_df['id'].repeat(len(nurse_df)).values

    scored_df = score_shifts(pairs_df, hospital_ids)

    scored_df['id_nurse'] = pairs_df['id_nurse']
    scored_df['id_shift'] = pairs_df['id_shift']

    return scored_df.sort_values("pred_score", ascending=False).to_dict(orient="records")

@app.post("/retrain-model")
def train():
    train_model()
    return {"message": "Model retrained"}


