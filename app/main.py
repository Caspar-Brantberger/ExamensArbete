from fastapi import FastAPI, Depends, HTTPException, Header, Path
from sqlalchemy.orm import Session
from typing import Optional, List
from uuid import UUID
import base64
import os
import jwt
from dotenv import load_dotenv
import pandas as pd

from app.db.database import get_db
from app.models.nurse import Nurse
from app.models.shift_advertisement import ShiftAdvertisement
from app.ML.inference.score_shifts import score_shifts
from app.ML.models import load_model
from app.ML.model_loader import init as init_model
from app.ML.build_pairs_with_features import build_pairs_with_features
from app.schemas import NurseWithShiftsSchema, ShiftSchema, NurseBaseSchema
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="PH Suggestion Engine")

# =========================
# Startup event: init ML model
# =========================
@app.on_event("startup")
def startup_event():
    init_model()


# =========================
# Config / JWT
# =========================
SECRET_KEY = base64.b64decode(os.getenv("ISSUER_SECRET"))
ALGORITHM = "HS512"

def verify_token(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


# =========================
# Public, production-facing endpoints
# =========================
@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "model": "loaded"
    }

# --- Request/response models for core endpoint ---
class CandidateShift(BaseModel):
    id: str = Field(..., alias="id")   # shift id

    class Config:
        allow_population_by_field_name = True

class NurseRef(BaseModel):
    id: str = Field(..., alias="id")
    class Config:
        allow_population_by_field_name = True

class RecommendShiftsRequestCandidates(BaseModel):
    nurse: NurseRef
    candidate_shifts: List[CandidateShift]

class RecommendShiftsRequestFromDB(BaseModel):
    nurse_id: str
    max_results: Optional[int] = 20

class RecommendationItem(BaseModel):
    shift_id: str
    score: float

class RecommendShiftsResponse(BaseModel):
    nurse_id: str
    recommendations: List[RecommendationItem]


@app.post("/v1/recommendations/shifts")
def recommend_shifts(
    request: dict, 
    db: Session = Depends(get_db),
    authorization: str = Header(None)
):
    """
    Main scoring endpoint for Spring Boot.
    Request can be either:
    - { "nurse_id": "...", "max_results": 20 }
    - { "nurse": {...}, "candidate_shifts": [...] }
    """
    token_payload = verify_token(authorization)

    # Fetch nurse from DB if nurse_id provided
    nurse_id = request.get("nurse_id") or request.get("nurse", {}).get("id")
    if not nurse_id:
        raise HTTPException(status_code=400, detail="nurse_id is required")

    nurse = db.query(Nurse).filter(Nurse.id == nurse_id).first()
    if not nurse:
        raise HTTPException(status_code=404, detail="Nurse not found")

    # Fetch shifts
    candidate_shifts = request.get("candidate_shifts")
    if candidate_shifts:
        shift_ids = [s["id"] for s in candidate_shifts]
        shifts = db.query(ShiftAdvertisement).filter(ShiftAdvertisement.id.in_(shift_ids)).all()
    else:
        shifts = db.query(ShiftAdvertisement).all()

    # Build scoring DataFrames
    nurse_df = pd.DataFrame([nurse.__dict__])
    shifts_df = pd.DataFrame([s.__dict__ for s in shifts])
    nurse_shifts_df = pd.DataFrame(columns=['nurse_id', 'shift_id'])
    pairs_df, hospital_ids = build_pairs_with_features(nurse_df, shifts_df, nurse_shifts_df)

    pairs_df['id_nurse'] = nurse_df['id'].iloc[0]
    pairs_df['id_shift'] = shifts_df['id'].iloc[0] if len(shifts_df) == 1 else shifts_df['id'].repeat(len(nurse_df)).values

    scored_df = score_shifts(pairs_df, hospital_ids)
    scored_df['id_nurse'] = pairs_df['id_nurse']
    scored_df['id_shift'] = pairs_df['id_shift']

    # Limit results if requested
    max_results = request.get("max_results")
    if max_results:
        scored_df = scored_df.nlargest(max_results, "pred_score")

    recommendations = [
        {"shift_id": r["id_shift"], "score": float(r["pred_score"])}
        for r in scored_df.to_dict(orient="records")
    ]

    return {"nurse_id": nurse_id, "recommendations": recommendations}


# =========================
# Internal / dev endpoints
# =========================
@app.get("/internal/dev-data/{nurse_id}", response_model=NurseWithShiftsSchema)
def get_dev_data(nurse_id: str, db: Session = Depends(get_db)):
    nurse = db.query(Nurse).filter(Nurse.id == nurse_id).first()
    if not nurse:
        raise HTTPException(status_code=404, detail="Nurse not found")
    return NurseWithShiftsSchema.from_orm(nurse)


@app.get("/internal/nurses", response_model=list[NurseBaseSchema])
def get_all_nurses(db: Session = Depends(get_db)):
    nurses = db.query(Nurse).all()
    return [NurseBaseSchema.from_orm(s) for s in nurses]


@app.get("/internal/shifts", response_model=list[ShiftSchema])
def get_all_shifts(db: Session = Depends(get_db)):
    shifts = db.query(ShiftAdvertisement).all()
    return [ShiftSchema.from_orm(s) for s in shifts]


@app.get("/internal/recommend/{nurse_id}")
def internal_recommend(nurse_id: str, db: Session = Depends(get_db)):
    # Placeholder internal recommendation
    return {"nurse_id": nurse_id, "recommendations": ["shift-1", "shift-2", "shift-3"]}