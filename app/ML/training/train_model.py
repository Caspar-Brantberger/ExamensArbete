def train_model():
    import pandas as pd
    import numpy as np
    import joblib
    import lightgbm as lgb
    from sqlalchemy import create_engine
    from pathlib import Path
    from app.ML.build_pairs_with_features import build_pairs_with_features
    import os

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")

    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:5432/{DB_NAME}")

    nurses = pd.read_sql("SELECT * FROM nurse", engine)
    shifts = pd.read_sql("SELECT * FROM shift_advertisement", engine)
    nurse_shifts = pd.read_sql("SELECT * FROM nurse_shift", engine)

    df_pairs = build_pairs_with_features(nurses, shifts, nurse_shifts)

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

    feature_columns = [col for col in df_pairs.columns if col != 'label']

    X = df_pairs[feature_columns].fillna(0)
    y = df_pairs['label']

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    model.fit(X, y)

    model_path = Path(__file__).parent / "models/nurse_shift_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model()