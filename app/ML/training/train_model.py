import pandas as pd
import numpy as np
import ast
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from geopy.distance import geodesic

# =========================
# 0. Notes for real data usage
# =========================
"""
This script is a prototype for training a LightGBM model to predict nurse-shift matching scores.
Currently, it uses synthetic or placeholder data for development purposes.

When using real data, make sure to update:

1. Database connection:
- Replace 'database', 'USERNAME', 'PASSWORD' with your actual PostgreSQL credentials.
    Example:
    engine = create_engine("postgresql://USERNAME:PASSWORD@HOST:PORT/DATABASE")

2. Table names:
- Ensure tables exist: shift_advertisement, nurse, nurse_shift
- Adjust SQL queries if your schema differs

3. Placeholder values:
- Lat/lng, experience, lead_time, hourly rates, hospital familiarity, etc. are random.
- Replace with real values or computed features from your historical data

4. Target:
- Currently random (0/1)
- Replace with your real label (e.g., whether nurse accepted the shift or historical match score)

5. Categorical features:
- Using LightGBM categorical encoding (.astype('category').cat.codes)
- Adjust as needed for real data (LightGBM supports categories directly)

6. Other synthetic features:
- total_shifts_completed, avg_distance_accepted, night_shift_preference, hospital familiarity
- Replace with historical metrics if available
"""

# =========================
# 1. Load data from database
# =========================
# Replace with your actual credentials
engine = create_engine("postgresql://USERNAME:PASSWORD@localhost:5432/DATABASE")

shifts = pd.read_sql("SELECT * FROM shift_advertisement", engine)
nurses = pd.read_sql("SELECT * FROM nurse", engine)
nurse_shifts = pd.read_sql("SELECT * FROM nurse_shift", engine)

# =========================
# 2. Preprocess nurses and shifts
# =========================
today = pd.Timestamp('2025-12-01')
nurses['date_of_birth'] = pd.to_datetime(nurses['date_of_birth'], errors='coerce')
nurses['age'] = ((today - nurses['date_of_birth']).dt.days // 365).fillna(30).astype(int)

# Lat/lng - placeholder values
shifts['lat'] = shifts['lat'].fillna(pd.Series(np.random.uniform(55.3, 69.1, len(shifts)), index=shifts.index))
shifts['lng'] = shifts['lng'].fillna(pd.Series(np.random.uniform(11.1, 24.2, len(shifts)), index=shifts.index))

# Nurses base location - placeholder
nurses['base_lat'] = np.random.uniform(55.3, 69.1, len(nurses))
nurses['base_lng'] = np.random.uniform(11.1, 24.2, len(nurses))

# Preferred locations as list
nurses['preferred_locations'] = nurses.get('preferred_locations', '[]')
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

# Synthetic historical/aggregated fields - replace with real metrics
nurses['total_shifts_completed'] = np.random.randint(0, 100, len(nurses))
nurses['avg_distance_accepted'] = np.random.uniform(0, 200, len(nurses))
nurses['night_shift_preference'] = np.random.uniform(0, 1, len(nurses))
nurses['avg_hourly_rate_accepted'] = np.random.uniform(200, 600, len(nurses))

# Hospital familiarity: simulated as random integers - replace with real history
hospital_ids = shifts['hospital_id'].unique()
for hid in hospital_ids:
    nurses[f'hospital_{hid}_familiarity'] = np.random.randint(0, 10, len(nurses))

# =========================
# 3. Merge historical nurse-shift data if available
# =========================
if not nurse_shifts.empty:
    # Average distance accepted
    if 'distance_km' in nurse_shifts.columns:
        avg_dist = nurse_shifts.groupby('nurse_id')['distance_km'].mean().rename('avg_distance_accepted')
        nurses = nurses.merge(avg_dist, left_on='id', right_index=True, how='left')
        nurses['avg_distance_accepted'] = nurses['avg_distance_accepted'].fillna(np.random.uniform(0,200,len(nurses)))
    # Night shift preference
    if 'is_night_shift' in nurse_shifts.columns:
        night_pref = nurse_shifts.groupby('nurse_id')['is_night_shift'].mean().rename('night_shift_preference')
        nurses = nurses.merge(night_pref, left_on='id', right_index=True, how='left')
        nurses['night_shift_preference'] = nurses['night_shift_preference'].fillna(np.random.uniform(0,1,len(nurses)))
    # Avg hourly rate
    if 'hourly_rate' in nurse_shifts.columns:
        avg_rate = nurse_shifts.groupby('nurse_id')['hourly_rate'].mean().rename('avg_hourly_rate_accepted')
        nurses = nurses.merge(avg_rate, left_on='id', right_index=True, how='left')
        nurses['avg_hourly_rate_accepted'] = nurses['avg_hourly_rate_accepted'].fillna(np.random.uniform(200,600,len(nurses)))

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
    return geodesic((row['base_lat'], row['base_lng']), (row['lat'], row['lng'])).km
pairs['distance_km'] = pairs.apply(compute_distance, axis=1)

pairs['specialization_match'] = (pairs['specialization_nurse'] == pairs['specialization_shift']).astype(int)
pairs['role_match'] = (pairs['role_id'] == pairs['role_id']).astype(int)  # Placeholder

# Location preference
def location_match(row):
    locs = row.get('preferred_locations', [])
    if locs is None: locs = []
    return int(row['location'] in locs)
pairs['location_preference_match'] = pairs.apply(location_match, axis=1)

# Shift features
pairs['shift_start_date'] = pd.to_datetime(pairs['shift_start_date'])
pairs['shift_hour'] = pairs['shift_start_date'].dt.hour
pairs['is_night_shift'] = ((pairs['shift_hour'] >= 20) | (pairs['shift_hour'] < 6)).astype(int)
pairs['lead_time_days'] = np.random.randint(1, 30, len(pairs))  # Placeholder
pairs['experience_gap'] = np.random.randint(-5,5,len(pairs))     # Placeholder

# =========================
# 6. Encode categorical features for LightGBM
# =========================
for col in ['specialization_nurse', 'role_id', 'shift_type', 'location']:
    pairs[col] = pairs[col].astype('category').cat.codes

# =========================
# 7. Target (placeholder)
# =========================
pairs['target'] = np.random.randint(0,2,len(pairs))

# =========================
# 8. Train-test split
# =========================
feature_cols = [
    'age', 'distance_km', 'specialization_match', 'role_match',
    'experience_gap', 'is_night_shift', 'lead_time_days', 'location_preference_match',
    'avg_distance_accepted', 'night_shift_preference', 'avg_hourly_rate_accepted'
] + [f'hospital_{hid}_familiarity' for hid in hospital_ids]

X = pairs[feature_cols]
y = pairs['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# 9. Train LightGBM model
# =========================
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt'
}

model = lgb.train(params, lgb_train, valid_sets=[lgb_train,lgb_test], num_boost_round=100)

# =========================
# 10. Predictions
# =========================
pairs['pred_score'] = model.predict(X)
top_matches = pairs.sort_values('pred_score', ascending=False).groupby('id_shift').head(5)
print(top_matches[['id_nurse','id_shift','pred_score']])