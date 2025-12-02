import pandas as pd
from app.ML.models import load_model
"""
To use with real data:
- Fetch shifts and nurses from the database.
- Create nurse-shift pairs as in train_model.py.
- Perform feature engineering:
    - distance_km using geodesic
    - specialization_match, role_match, location_preference_match
    - is_night_shift, lead_time_days, experience_gap
    - historical features: avg_distance_accepted, night_shift_preference, avg_hourly_rate_accepted, hospital_{id}_familiarity
- Convert categorical columns to numeric using cat.codes
- Ensure that feature_cols exactly match the features used to train the model.
"""

model = load_model()

def score_shifts(pairs_df: pd.DataFrame) -> pd.DataFrame:
    
    feature_cols = [col for col in pairs_df.columns if col not in ['id_nurse', 'id_shift']]
    X = pairs_df[feature_cols]

    pairs_df['pred_score'] = model.predict(X)
    return pairs_df