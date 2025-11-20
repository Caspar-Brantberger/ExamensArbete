from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from .associations import nurse_shift
from app.db.database import Base  

class Nurse(Base):
    __tablename__ = "nurse"
    id = Column(String, primary_key=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    shifts = relationship("ShiftAdvertisement", secondary=nurse_shift, back_populates="nurses")