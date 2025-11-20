from fastapi import FastAPI,Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.nurse import Nurse


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/dev-data/{nurse_id}")
def get_dev_data(nurse_id: str, db: Session = Depends(get_db)):
    nurse = db.query(Nurse).filter(Nurse.id == nurse_id).first()
    if not nurse:
        return {"nurse": None}
    return {"nurse": {
        "id": nurse.id,
        "first_name": nurse.first_name,
        "last_name": nurse.last_name
    }}