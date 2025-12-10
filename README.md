# PH Suggestion Engine - Dev Data API

## Schema Assumptions
- Nurse:
  - `id`: UUID
  - `first_name`, `last_name`: str
  - `location`: optional str
  - `skills`: optional list[str]
- Shift:
  - `id`: UUID
  - `unique_shift_id`: str
  - `hospital_name`: str
  - `description`: optional str
  - `shift_start_date`, `shift_end_date`: optional datetime
  - `experience`, `location`, `shift_type`: optional str

## API Endpoints
- GET `/dev-data/{nurse_id}` → returns nurse + their shifts
- GET `/nurses` → returns list of all nurses (id, name, location)
- GET `/all-shifts` → returns list of all shifts
- POST `/assign-shift/{nurse_id}/{shift_id}` → assign shift to nurse

## Next Steps for Feature Engineering
- Map nurse skills, availability, and shift requirements into feature vectors
- Normalize date/times and categorical variables
- Ensure UUID consistency when generating features