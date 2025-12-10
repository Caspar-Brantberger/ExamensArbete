from sqlalchemy import Table, Column, String, ForeignKey
from app.db.database import Base

nurse_shift = Table(
    "nurse_shift",
    Base.metadata,
    Column("nurse_id", String, ForeignKey("nurse.id"), primary_key=True),
    Column("shift_id", String, ForeignKey("shift_advertisement.id"), primary_key=True)
)
