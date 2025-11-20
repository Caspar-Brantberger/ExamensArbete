from fastapi import FastAPI,Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.nurse import Nurse
from app.models.shift_advertisement import ShiftAdvertisement
from app.models.associations import nurse_shift



app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/dev-data/{nurse_id}")
def get_dev_data(nurse_id: str, db: Session = Depends(get_db)):
    nurse = db.query(Nurse).filter(Nurse.id == nurse_id).first()
    if not nurse:
        return{"nurse":None,"shifts":[]}

    return{
        "nurse":{
            "id":nurse.id,
            "first_name":nurse.first_name,
            "last_name":nurse.last_name,
        },
        "shifts":[
            {
                "id": s.id,
                "unique_shift_id": s.unique_shift_id,
                "hospital_name": s.hospital_name,
                "description": s.description,
                "shift_start_date": s.shift_start_date,
                "shift_end_date": s.shift_end_date,
                "experience": s.experience,
                "location": s.location,
                "shift_type": s.shift_type,
                
            } 
            for s in nurse.shifts
        ]
    }

@app.post("/assign-shift/{nurse_id}/{shift_id}")
def assign_shift_to_nurse(nurse_id: str, shift_id: str, db: Session = Depends(get_db)):
    db.execute(nurse_shift.insert().values(nurse_id=nurse_id, shift_id=shift_id))
    db.commit()
    return {"message": f"Shift {shift_id} assigned to Nurse {nurse_id}" }


@app.get("/all-shifts")
def get_all_shifts(db: Session = Depends(get_db)):
    shifts = db.query(ShiftAdvertisement).all()
    return[
        {
            "id": s.id,
            "unique_shift_id": s.unique_shift_id,
            "hospital_name": s.hospital_name,
        }
        for s in shifts
    ]

