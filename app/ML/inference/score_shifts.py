import pandas as pd
import numpy as np
from geopy.distance import geodesic
from typing import List
from app.ML.model_loader import get_model
import lightgbm as lgb
from uuid import UUID

"""
This function scores nurse-shift pairs using the trained ML model.

When using real production data:
1. Retrieve shifts and nurses from the database.
2. Generate all possible nurse-shift combinations (same process as in train_model.py).
3. Perform feature engineering to recreate the exact features used during training, e.g.:
    - distance_km (computed via geodesic distance)
    - specialization_match, role_match, location_preference_match
    - is_night_shift, lead_time_days, experience_gap
    - historical features:
        * avg_distance_accepted
        * night_shift_preference
        * avg_hourly_rate_accepted
        * hospital_{id}_familiarity
4. Convert categorical columns to numeric using .cat.codes or a consistent encoding method.
5. Ensure that feature_cols matches the training feature set in the same order.
The model will not work correctly if features are missing or misordered.
"""


FEATURE_COLS_BASE = [
    'age', 'distance_km', 'specialization_match', 'role_match',
    'experience_gap', 'is_night_shift', 'lead_time_days', 'location_preference_match',
    'avg_distance_accepted', 'night_shift_preference', 'avg_hourly_rate_accepted',
    'shift_hour', 'shift_day_of_week', 'shift_duration_hours', 'preferred_location_distance_rank'
]

def safe_distance(row):
    if pd.isna(row['base_lat']) or pd.isna(row['base_lng']) or pd.isna(row['lat']) or pd.isna(row['lng']):
        return 0.0  # SYNTHETIC fallback
    return geodesic((row['base_lat'], row['base_lng']), (row['lat'], row['lng'])).km

def ensure_features(df: pd.DataFrame, hospital_ids: list[UUID]) -> pd.DataFrame:
    for col in FEATURE_COLS_BASE:
        if col not in df.columns:
            if col in ['distance_km', 'avg_distance_accepted', 'experience_gap', 'shift_duration_hours',
                    'lead_time_days', 'preferred_location_distance_rank']:
                df[col] = 0
            elif col in ['age']:
                df[col] = 30
            elif col in ['night_shift_preference', 'location_preference_match', 'specialization_match',
                        'role_match']:
                df[col] = 0
            elif col in ['avg_hourly_rate_accepted', 'shift_hour', 'shift_day_of_week']:
                df[col] = 200

    for hid in hospital_ids:
        hid_str = str(hid).replace('-', '_')
        col_name = f'hospital_{hid_str}_familiarity'
        if col_name not in df.columns:
            df[col_name] = 0

    return df

def score_shifts(pairs_df: pd.DataFrame, hospital_ids: list) -> pd.DataFrame:

    model = get_model()
    pairs_df = ensure_features(pairs_df.copy(), hospital_ids)

    
    feature_cols = FEATURE_COLS_BASE + [f"hospital_{str(hid).replace('-', '_')}_familiarity" for hid in hospital_ids]
    X = pairs_df[feature_cols]

    if(isinstance(model,lgb.Booster)):
        pairs_df['pred_score'] = model.predict(X)
    else:
        pairs_df['pred_score'] = model.predict(X)

    return pairs_df