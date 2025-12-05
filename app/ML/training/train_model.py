def train_model():
    import pandas as pd
    import numpy as np
    from datetime import date
    import joblib
    import lightgbm as lgb
    from sqlalchemy import create_engine
    from pathlib import Path
    from app.ML.build_pairs_with_features import build_pairs_with_features
    import os
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Load database credentials from environment variables
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")
    
    # Ensure credentials are set
    if not all([DB_USER, DB_PASSWORD, DB_NAME]):
        raise ValueError("Database credentials are not fully set in environment variables.")

    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:5432/{DB_NAME}")

    nurses = pd.read_sql("SELECT * FROM nurse", engine)
    shifts = pd.read_sql("SELECT * FROM shift_advertisement", engine)
    nurse_shifts = pd.read_sql("SELECT * FROM nurse_shift", engine)

    # Build pairwise features
    # TODO: Once we have more real nurse/shift data, check that all relevant features are include
    df_pairs, hospital_ids = build_pairs_with_features(nurses, shifts, nurse_shifts)
    
    # Base features to use in model
    FEATURE_COLS_BASE = [
    'age', 'distance_km', 'specialization_match', 'role_match',
    'experience_gap', 'is_night_shift', 'lead_time_days', 'location_preference_match',
    'avg_distance_accepted', 'night_shift_preference', 'avg_hourly_rate_accepted',
    'shift_hour', 'shift_day_of_week', 'shift_duration_hours', 'preferred_location_distance_rank'
    ]
    
    # Drop sensitive / irrelevant columns for modeling
    # TODO: Ensure this matches real database schema
    df_pairs = df_pairs.drop(columns=[
    'first_name', 'last_name', 'social_security_number', 'street_address',
    'postal_code', 'city', 'email', 'password', 'bio', 'bank_account_id'
    ], errors='ignore')

    
    # Count preferred locations (temporary feature for model)
    # TODO: In real data, preferred_locations may be used differently
    df_pairs['preferred_locations_count'] = df_pairs['preferred_locations'].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    if 'preferred_locations' in df_pairs.columns:
        df_pairs = df_pairs.drop(columns=['preferred_locations'])

    if 'label' not in df_pairs.columns:
        df_pairs['label'] = 0

    for col in ['specialization_nurse', 'specialization_shift', 'shift_type', 'location', 'hospital_id']:
        if col in df_pairs.columns:
            df_pairs[col], _ = pd.factorize(df_pairs[col])

    feature_columns = FEATURE_COLS_BASE + [f"hospital_{hid}_familiarity" for hid in hospital_ids]

    X = df_pairs[feature_columns].astype(float)
    y = df_pairs['label'].astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    model.fit(X, y)

    model_path = Path(__file__).parent / f"models/nurse_shift_model_{date.today()}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model()