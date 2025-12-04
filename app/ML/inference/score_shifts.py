import pandas as pd
from app.ML.models import load_model

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

model = load_model()

FEATURE_COLS_BASE = [
    'age', 'distance_km', 'specialization_match', 'role_match',
    'experience_gap', 'is_night_shift', 'lead_time_days', 'location_preference_match',
    'avg_distance_accepted', 'night_shift_preference', 'avg_hourly_rate_accepted',
    'shift_hour', 'shift_day_of_week', 'shift_duration_hours', 'preferred_location_distance_rank'
]

def ensure_features(df: pd.DataFrame, hospital_ids: list):
    for col in FEATURE_COLS_BASE:
        if col not in df.columns:
            if col in ['distance_km', 'avg_distance_accepted', 'experience_gap', 'shift_duration_hours',
                    'lead_time_days', 'preferred_location_distance_rank']:
                df[col] = np.random.uniform(0, 10, len(df))
            elif col in ['age']:
                df[col] = 30
            elif col in ['night_shift_preference', 'location_preference_match', 'specialization_match',
                        'role_match']:
                df[col] = 0
            elif col in ['avg_hourly_rate_accepted', 'shift_hour', 'shift_day_of_week']:
                df[col] = np.random.uniform(200, 600, len(df))

    for hid in hospital_ids:
        col_name = f'hospital_{hid}_familiarity'
        if col_name not in df.columns:
            df[col_name] = np.random.randint(0, 10, len(df))

    return df

def score_shifts(pairs_df: pd.DataFrame, hospital_ids: list) -> pd.DataFrame:
    

    pairs_df = ensure_features(pairs_df, hospital_ids)

    pairs_df['distance_km'] = pairs_df['distance_km'].fillna(0)
    pairs_df['age'] = pairs_df['age'].fillna(30)

    feature_cols = FEATURE_COLS_BASE + [f'hospital_{hid}_familiarity' for hid in hospital_ids]
    X = pairs_df[feature_cols]

    pairs_df['pred_score'] = model.predict(X)
    return pairs_df