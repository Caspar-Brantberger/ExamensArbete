from __future__ import annotations
from uuid import UUID
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class ShiftSchema(BaseModel):
    id: UUID
    unique_shift_id: str
    hospital_name: Optional[str] = None
    description: Optional[str] = None
    shift_start_date: Optional[datetime] = None
    shift_end_date: Optional[datetime] = None
    experience: Optional[str] = None
    location: Optional[str] = None
    shift_type: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)