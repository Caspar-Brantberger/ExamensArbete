from fastapi import FastAPI,Depends,HTTPException, Header, HTTPException, Request, Path
from uuid import UUID
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.nurse import Nurse
from app.models.shift_advertisement import ShiftAdvertisement
from app.models.associations import nurse_shift
from app.schemas import ShiftSchema
from app.schemas import NurseWithShiftsSchema
from app.schemas import NurseBaseSchema
from pydantic import BaseModel
import jwt
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


app = FastAPI()

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


load_dotenv()

app = FastAPI()

SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS512"

print(f"=== FastAPI Starting ===")
print(f"Algorithm: {ALGORITHM}")
print(f"SECRET_KEY loaded: {'YES' if SECRET_KEY else 'NO'}")


class NurseInfo(BaseModel):
    nurse_id: str
    name: str
    specialization: str
    experience_years: int


class Shift(BaseModel):
    shift_id: str
    hospital_id: str
    start_time: str
    end_time: str
    department: str
    required_specialization: str


class RecommendationRequest(BaseModel):
    nurse: NurseInfo
    shifts: List[Shift]


class RecommendationResponse(BaseModel):
    shift_ids: List[str]
    received_nurse: str


def verify_token(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    token = authorization.split(" ")[1]
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print(f"✓ Token verified for user: {payload.get('sub')}")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


@app.post("/recommend", response_model=RecommendationResponse)
def recommend_post(
    request: RecommendationRequest, 
    authorization: str = Header(None)
):
    """POST endpoint - Receives full nurse and shift data"""
    token_payload = verify_token(authorization)
    
    print(f"\n=== POST /recommend ===")
    print(f"Nurse: {request.nurse.name} (ID: {request.nurse.nurse_id})")
    print(f"Specialization: {request.nurse.specialization}")
    print(f"Number of shifts: {len(request.shifts)}")
    
    # Your recommendation logic here
    # For now, returning mock recommendations
    recommended_shifts = []
    for shift in request.shifts:
        # Simple matching logic - recommend if specializations match
        if shift.required_specialization == request.nurse.specialization:
            recommended_shifts.append(shift.shift_id)
    
    # If no matches, recommend first 3 shifts
    if not recommended_shifts:
        recommended_shifts = [s.shift_id for s in request.shifts[:3]]
    
    return RecommendationResponse(
        shift_ids=recommended_shifts,
        received_nurse=request.nurse.nurse_id
    )


@app.get("/recommend/{nurse_id}", response_model=RecommendationResponse)
def recommend_get(
    nurse_id: str = Path(..., description="The nurse ID"),
    authorization: str = Header(None)
):
    """GET endpoint - Simple nurse ID lookup"""
    token_payload = verify_token(authorization)
    
    print(f"\n=== GET /recommend/{nurse_id} ===")
    print(f"Looking up recommendations for nurse: {nurse_id}")
    
    # Your lookup logic here
    # For now, returning mock data
    return RecommendationResponse(
        shift_ids=["shift-1", "shift-2", "shift-3"],
        received_nurse=nurse_id
    )