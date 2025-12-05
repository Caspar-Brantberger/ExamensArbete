import pandas as pd
import numpy as np
import ast
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from geopy.distance import geodesic
import joblib

# =========================
# 0. Notes for real data usage
# =========================
"""
This script trains a prototype LightGBM model for nurse-shift matching.

Current version:
- Uses synthetic / placeholder values for many fields (distance, experience, lead time, historical stats, etc.)
- Demonstrates the full ML process: loading data → expanding pairs → feature engineering → training → scoring
- Allows experimenting with the scoring logic before real historical data is available.

Why synthetic data?
    
The project was missing several important columns and historical values required by the model (e.g.,
avg_distance_accepted, night_shift_preference, hospital_familiarity, lat/lng, experience_gap, etc.). 
Without these, it's not possible to train a complete ML model.

To be able to:

test the full ML pipeline and feature engineering,

verify the integration from backend → ML → scoring,

build a functional prototype at this stage,

I used synthetic values for the features that do not exist yet.

Synthetic data was not used to create an accurate model, but to validate that the structure, logic, and system flow work as intended. 
Once real data becomes available, all synthetic values will be replaced and the model can be retrained without any code changes.

When integrating real production data:
1. Replace database credentials and ensure tables match your schema.
2. Replace all placeholder features (random values) with actual engineered metrics:
    - true historical acceptance rates
    - real distance values
    - real experience gap
    - real lead-time between posting and shift start
3. Recompute historical aggregated nurse features from nurse_shift history.
4. Keep Feature Engineering identical between train and inference.
5. Replace the synthetic target with your real target label.
6. Re-train and compare multiple ML algorithms (LightGBM, Random Forest, XGBoost, Logistic Regression).
7. Select the best model based on validation metrics and real-world behavior.
"""


# =========================
# 1. Load data
# =========================
engine = create_engine("postgresql://localhost:5432/Database?user=username&password=password")

shifts = pd.read_sql("SELECT * FROM shift_advertisement", engine)
nurses = pd.read_sql("SELECT * FROM nurse", engine)
nurse_shifts = pd.read_sql("SELECT * FROM nurse_shift", engine)

# =========================
# 2. Fill missing / synthetic data
# =========================
today = pd.Timestamp('2025-12-01')
nurses['date_of_birth'] = pd.to_datetime(nurses['date_of_birth'], errors='coerce')
nurses['age'] = ((today - nurses['date_of_birth']).dt.days // 365).fillna(30).astype(int)  # SYNTHETIC default age

# Ensure role_id exists
if 'role_id' not in nurses.columns:
    nurses['role_id'] = 0  # SYNTHETIC default
if 'role_id' not in shifts.columns:
    shifts['role_id'] = 0  # SYNTHETIC default

# Lat/lng fallback for shifts
shifts['lat'] = shifts['lat'].fillna(pd.Series(np.random.uniform(55.3, 69.1, len(shifts)), index=shifts.index))  # SYNTHETIC
shifts['lng'] = shifts['lng'].fillna(pd.Series(np.random.uniform(11.1, 24.2, len(shifts)), index=shifts.index))  # SYNTHETIC

# Base location for nurses
nurses['base_lat'] = np.random.uniform(55.3, 69.1, len(nurses))  # SYNTHETIC
nurses['base_lng'] = np.random.uniform(11.1, 24.2, len(nurses))  # SYNTHETIC

# Preferred locations
nurses['preferred_locations'] = nurses.get('preferred_locations', '[]')  # SYNTHETIC fallback

import ast
def safe_literal_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    elif isinstance(x, list):
        return x
    else:
        return []
nurses['preferred_locations'] = nurses['preferred_locations'].apply(safe_literal_eval)

# Synthetic historical features
nurses['total_shifts_completed'] = np.random.randint(0, 100, len(nurses))  # SYNTHETIC
nurses['avg_distance_accepted'] = np.random.uniform(0, 200, len(nurses))  # SYNTHETIC
nurses['night_shift_preference'] = np.random.uniform(0, 1, len(nurses))  # SYNTHETIC
nurses['avg_hourly_rate_accepted'] = np.random.uniform(200, 600, len(nurses))  # SYNTHETIC

# Hospital familiarity: synthetic per nurse per hospital
hospital_ids = shifts['hospital_id'].unique()
for hid in hospital_ids:
    nurses[f'hospital_{hid}_familiarity'] = np.random.randint(0, 10, len(nurses))  # SYNTHETIC

# =========================
# 3. Merge historical nurse_shifts (if available)
# =========================
if not nurse_shifts.empty:
    if 'distance_km' in nurse_shifts.columns:
        avg_dist = nurse_shifts.groupby('nurse_id')['distance_km'].mean().rename('avg_distance_accepted')
        nurses = nurses.merge(avg_dist, left_on='id', right_index=True, how='left')
        nurses['avg_distance_accepted'] = nurses['avg_distance_accepted'].fillna(np.random.uniform(0,200,len(nurses)))  # SYNTHETIC fallback
    if 'is_night_shift' in nurse_shifts.columns:
        night_pref = nurse_shifts.groupby('nurse_id')['is_night_shift'].mean().rename('night_shift_preference')
        nurses = nurses.merge(night_pref, left_on='id', right_index=True, how='left')
        nurses['night_shift_preference'] = nurses['night_shift_preference'].fillna(np.random.uniform(0,1,len(nurses)))  # SYNTHETIC fallback
    if 'hourly_rate' in nurse_shifts.columns:
        avg_rate = nurse_shifts.groupby('nurse_id')['hourly_rate'].mean().rename('avg_hourly_rate_accepted')
        nurses = nurses.merge(avg_rate, left_on='id', right_index=True, how='left')
        nurses['avg_hourly_rate_accepted'] = nurses['avg_hourly_rate_accepted'].fillna(np.random.uniform(200,600,len(nurses)))  # SYNTHETIC fallback

# =========================
# 4. Expand nurse-shift pairs
# =========================
pairs = pd.merge(
    nurses.assign(key=1),
    shifts.assign(key=1),
    on='key',
    suffixes=('_nurse', '_shift')
)
pairs.drop('key', axis=1, inplace=True)

# =========================
# 5. Feature engineering
# =========================
def compute_distance(row):
    if pd.notnull(row['base_lat']) and pd.notnull(row['base_lng']) and pd.notnull(row['lat']) and pd.notnull(row['lng']):
        return geodesic((row['base_lat'], row['base_lng']), (row['lat'], row['lng'])).km
    else:
        return np.nan
pairs['distance_km'] = pairs.apply(compute_distance, axis=1).fillna(0)  # SYNTHETIC fallback

# Pair features
pairs['specialization_match'] = (pairs['specialization_nurse'] == pairs['specialization_shift']).astype(int)
pairs['role_match'] = (pairs['role_id_nurse'] == pairs['role_id_shift']).astype(int)  # SYNTHETIC placeholder if always 1
pairs['location_preference_match'] = pairs.apply(lambda r: int(r['location'] in r.get('preferred_locations', [])), axis=1)

# Shift features
pairs['shift_start_date'] = pd.to_datetime(pairs['shift_start_date'])
pairs['shift_end_date'] = pd.to_datetime(pairs['shift_end_date'], errors='coerce')
pairs['shift_hour'] = pairs['shift_start_date'].dt.hour
pairs['shift_day_of_week'] = pairs['shift_start_date'].dt.dayofweek
pairs['shift_duration_hours'] = ((pairs['shift_end_date'] - pairs['shift_start_date']).dt.total_seconds() / 3600).fillna(8)  # SYNTHETIC fallback
pairs['is_night_shift'] = ((pairs['shift_hour'] >= 20) | (pairs['shift_hour'] < 6)).astype(int)
pairs['lead_time_days'] = (pairs['shift_start_date'] - pd.to_datetime(pairs.get('advertisement_created_date', today))).dt.days.fillna(np.random.randint(1,30,len(pairs)))  # SYNTHETIC
pairs['experience_gap'] = np.random.randint(-5,5,len(pairs))  # SYNTHETIC
pairs['preferred_location_distance_rank'] = np.random.randint(1,5,len(pairs))  # SYNTHETIC

# =========================
# 6. Encode categorical features
# =========================
for col in ['specialization_nurse', 'role_id', 'shift_type', 'location']:
    if col in pairs.columns:
        pairs[col] = pairs[col].astype('category').cat.codes

# =========================
# 7. Target
# =========================
pairs['target'] = np.random.randint(0,2,len(pairs))  # SYNTHETIC target for training

# =========================
# 8. Train-test split
# =========================
feature_cols = [
    'age', 'distance_km', 'specialization_match', 'role_match',
    'experience_gap', 'is_night_shift', 'lead_time_days', 'location_preference_match',
    'avg_distance_accepted', 'night_shift_preference', 'avg_hourly_rate_accepted',
    'shift_hour', 'shift_day_of_week', 'shift_duration_hours',
    'preferred_location_distance_rank'
] + [f'hospital_{hid}_familiarity' for hid in hospital_ids]

X = pairs[feature_cols]
y = pairs['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# 9. Train LightGBM
# =========================
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt'
}

model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_test], num_boost_round=100)

# =========================
# 10. Save model
# =========================
import joblib
joblib.dump(model, 'nurse_shift_model.pkl')
print("Model saved as nurse_shift_model.pkl")

# =========================
# 11. Predictions (example)
# =========================
pairs['pred_score'] = model.predict(X)
top_matches = pairs.sort_values('pred_score', ascending=False).groupby('id_shift').head(5)
print(top_matches[['id_nurse','id_shift','pred_score']])