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

def score_shifts(pairs_df: pd.DataFrame) -> pd.DataFrame:
    
    feature_cols = [col for col in pairs_df.columns if col not in ['id_nurse', 'id_shift']]
    X = pairs_df[feature_cols]

    pairs_df['pred_score'] = model.predict(X)
    return pairs_df