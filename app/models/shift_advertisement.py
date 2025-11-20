from sqlalchemy import Column, String, Integer, DateTime,ForeignKey,Boolean,Float
from app.db.database import Base
from sqlalchemy.orm import relationship
from .associations import nurse_shift


class ShiftAdvertisement(Base):
    __tablename__  = "shift_advertisement"


    id = Column(String,primary_key=True,index=True)
    nurses = relationship("Nurse", secondary=nurse_shift, back_populates="shifts")
    unique_shift_id = Column(String, nullable=False)
    advertisement_end_time = Column(DateTime, nullable=True)
    hospital_name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    shift_start_date = Column(DateTime, nullable=True)
    shift_end_date = Column(DateTime, nullable=True)
    experience = Column(String, nullable=True)
    additional_notes = Column(String, nullable=True)
    hospital_id = Column(String, nullable=True)
    deleted_by = Column(String, nullable=True)
    deleted_on = Column(DateTime, nullable=True)
    is_fake = Column(Boolean, nullable=True, default=False)
    nr_of_pos = Column(Integer, nullable=True)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    extent = Column(String(50), nullable=True)
    shift_type = Column(String(50), nullable=False)
    location = Column(String(255), nullable=True)
    specialization = Column(String, nullable=True)