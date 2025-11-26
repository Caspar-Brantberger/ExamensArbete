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



