from .nurse import NurseBaseSchema, NurseWithShiftsSchema
from .shift import ShiftSchema

NurseWithShiftsSchema.model_rebuild()
ShiftSchema.model_rebuild()

__all__ = ["NurseBaseSchema", "NurseWithShiftsSchema", "ShiftSchema"]