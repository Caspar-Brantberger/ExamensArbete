from sqlalchemy import Column, Integer, String
from app.db.database import Base  

class Nurse(Base):
    __tablename__ = "nurse"
    id = Column(String, primary_key=True, index=True)
    first_name = Column(String)
    last_name = Column(String)