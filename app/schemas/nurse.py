from __future__ import annotations
from uuid import UUID
from typing import List, Optional
from pydantic import BaseModel, ConfigDict
from .shift import ShiftSchema

class NurseBaseSchema(BaseModel):
    first_name: str
    last_name: str
    location: Optional[str] = None
    skills: Optional[List[str]] = None
    id: UUID

    model_config = ConfigDict(from_attributes=True)

class NurseWithShiftsSchema(NurseBaseSchema):
    id: UUID
    shifts: List[ShiftSchema] = []

    model_config = ConfigDict(from_attributes=True)