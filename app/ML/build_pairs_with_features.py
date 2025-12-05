import pandas as pd
import numpy as np
from geopy.distance import geodesic

def safe_distance(row):
    if pd.isna(row.get('lat')) or pd.isna(row.get('lng')) or pd.isna(row.get('base_lat')) or pd.isna(row.get('base_lng')):
        return 0.0
    return geodesic((row['base_lat'], row['base_lng']), (row['lat'], row['lng'])).km

def build_pairs_with_features(nurses: pd.DataFrame, shifts: pd.DataFrame, nurse_shifts: pd.DataFrame) -> pd.DataFrame:
    nurses = nurses.copy()
    shifts = shifts.copy()
    
    for col, default in [
        ('base_lat', lambda n: np.random.uniform(55.3, 69.1, size=len(n))),
        ('base_lng', lambda n: np.random.uniform(11.1, 24.2, size=len(n))),
        ('date_of_birth', lambda n: pd.Timestamp('2000-01-01')),
        ('avg_distance_accepted', lambda n: np.random.uniform(0, 200, size=len(n))),
        ('night_shift_preference', lambda n: np.random.uniform(0, 1, size=len(n))),
        ('avg_hourly_rate_accepted', lambda n: np.random.uniform(200, 600, size=len(n))),
        ('preferred_locations', lambda n: [[] for _ in range(len(n))]),
        ('total_shifts_completed', lambda n: np.zeros(len(n)))
    ]:
        if col not in nurses.columns:
            nurses[col] = default(nurses)
        else:
            if col == 'date_of_birth':
                nurses[col] = pd.to_datetime(nurses[col], errors='coerce').fillna(default(nurses))
            elif col == 'preferred_locations':
                nurses[col] = nurses[col].apply(lambda x: x if isinstance(x, list) else [])
            else:
                nurses[col] = nurses[col].fillna(default(nurses))
    
    nurses['age'] = ((pd.Timestamp('today') - pd.to_datetime(nurses['date_of_birth'], errors='coerce')).dt.days // 365).fillna(30)

    for col, default in [
        ('lat', lambda s: np.random.uniform(55.3, 69.1, size=len(s))),
        ('lng', lambda s: np.random.uniform(11.1, 24.2, size=len(s))),
        ('experience_required', lambda s: np.zeros(len(s))),
        ('role_id', lambda s: np.zeros(len(s)))
    ]:
        if col not in shifts.columns:
            shifts[col] = default(shifts)
        else:
            shifts[col] = shifts[col].fillna(shifts[col].mean() if shifts[col].dtype in [np.float64, np.int64] else 0)

    pairs_df = nurses.assign(key=1).merge(shifts.assign(key=1), on='key').drop('key', axis=1)

    pairs_df['distance_km'] = pairs_df.apply(safe_distance, axis=1)

    pairs_df['specialization_nurse'] = pairs_df.get('specialization', '')
    pairs_df['specialization_shift'] = pairs_df.get('specialization_shift', pairs_df.get('specialization', ''))
    pairs_df['specialization_match'] = (pairs_df['specialization_nurse'] == pairs_df['specialization_shift']).astype(int)

    pairs_df['role_id_nurse'] = pairs_df.get('role_id', 0)
    pairs_df['role_id_shift'] = pairs_df.get('role_id_shift', pairs_df.get('role_id', 0))
    pairs_df['role_match'] = (pairs_df['role_id_nurse'] == pairs_df['role_id_shift']).astype(int)

    pairs_df['location_preference_match'] = pairs_df.apply(
        lambda r: int(r['location'] in (r['preferred_locations'] if isinstance(r['preferred_locations'], list) else [])), 
        axis=1
    )

    pairs_df['shift_start_date'] = pd.to_datetime(pairs_df.get('shift_start_date'), errors='coerce')
    pairs_df['shift_end_date'] = pd.to_datetime(pairs_df.get('shift_end_date'), errors='coerce')
    pairs_df['shift_hour'] = pairs_df['shift_start_date'].dt.hour.fillna(0)
    pairs_df['shift_day_of_week'] = pairs_df['shift_start_date'].dt.dayofweek.fillna(0)
    pairs_df['shift_duration_hours'] = ((pairs_df['shift_end_date'] - pairs_df['shift_start_date']).dt.total_seconds() / 3600).fillna(8)
    pairs_df['is_night_shift'] = ((pairs_df['shift_hour'] >= 20) | (pairs_df['shift_hour'] < 6)).astype(int)
    pairs_df['lead_time_days'] = (pairs_df['shift_start_date'] - pd.Timestamp('today')).dt.days.fillna(1)

    pairs_df['experience_gap'] = pairs_df['total_shifts_completed'].fillna(0) - pairs_df['experience_required'].fillna(0)
    pairs_df['preferred_location_distance_rank'] = 1

    hospital_ids = shifts['hospital_id'].dropna().unique().tolist()
    for hid in hospital_ids:
        col_name = f"hospital_{hid}_familiarity"
        if col_name not in pairs_df.columns:
            pairs_df[col_name] = 0

    return pairs_df, hospital_ids