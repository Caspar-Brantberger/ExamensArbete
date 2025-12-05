import pandas as pd
import numpy as np
from geopy.distance import geodesic
from typing import List
from uuid import UUID

def safe_float(value, default):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except:
        return float(default)

def safe_distance(row):
    if pd.isna(row['lat']) or pd.isna(row['lng']) or pd.isna(row['base_lat']) or pd.isna(row['base_lng']):
        return 0.0
    return geodesic((row['base_lat'], row['base_lng']), (row['lat'], row['lng'])).km

def build_pairs_with_features(nurse: dict, shifts: list[dict]) -> pd.DataFrame:
    

    shifts_df = pd.DataFrame([{
        "id_shift": str(s.get("id")),
        "specialization_shift": s.get("specialization"),
        "location": s.get("location"),
        "shift_start_date": pd.to_datetime(s.get("shift_start_date", pd.NaT), errors='coerce'),
        "shift_end_date": pd.to_datetime(s.get("shift_end_date", pd.NaT), errors='coerce'),
        "shift_type": s.get("shift_type"),
        "lat": safe_float(s.get("lat"), np.random.uniform(55.3, 69.1)),  # SYNTHETIC
        "lng": safe_float(s.get("lng"), np.random.uniform(11.1, 24.2)),  # SYNTHETIC
        "hospital_id": s.get("hospital_id"),
        "role_id_shift": int(s.get("role_id", 0)),
        "experience_required": pd.to_numeric(s.get("experience_required", 0), errors='coerce')
    } for s in shifts])

    nurse_df = pd.DataFrame([{
        "id_nurse": str(nurse.get("id")),
        "specialization_nurse": nurse.get("specialization"),
        "role_id_nurse": int(nurse.get("role_id", 0)),
        "base_lat": safe_float(nurse.get("base_lat"), np.random.uniform(55.3, 69.1)),  # SYNTHETIC
        "base_lng": safe_float(nurse.get("base_lng"), np.random.uniform(11.1, 24.2)),  # SYNTHETIC
        "age": ((pd.Timestamp('today') - pd.to_datetime(nurse.get("date_of_birth", pd.Timestamp('2000-01-01')), errors='coerce')).days // 365) or 30,
        "avg_distance_accepted": safe_float(nurse.get("avg_distance_accepted"), np.random.uniform(0, 200)),
        "night_shift_preference": safe_float(nurse.get("night_shift_preference"), np.random.uniform(0, 1)),
        "avg_hourly_rate_accepted": safe_float(nurse.get("avg_hourly_rate_accepted"), np.random.uniform(200, 600)),
        "preferred_locations": nurse.get("preferred_locations", []) or [],
        "total_shifts_completed": pd.to_numeric(nurse.get("total_shifts_completed", 0), errors='coerce') or 0
    }])

    pairs_df = nurse_df.assign(key=1).merge(shifts_df.assign(key=1), on='key').drop('key', axis=1)

    # Features
    pairs_df['distance_km'] = pairs_df.apply(safe_distance, axis=1)
    pairs_df['specialization_match'] = (pairs_df['specialization_nurse'] == pairs_df['specialization_shift']).astype(int)
    pairs_df['role_match'] = (pairs_df['role_id_nurse'] == pairs_df['role_id_shift']).astype(int)
    pairs_df['location_preference_match'] = pairs_df.apply(lambda r: int(r['location'] in r['preferred_locations']), axis=1)
    pairs_df['shift_start_date'] = pd.to_datetime(pairs_df['shift_start_date'])
    pairs_df['shift_end_date'] = pd.to_datetime(pairs_df['shift_end_date'], errors='coerce')
    pairs_df['shift_hour'] = pairs_df['shift_start_date'].dt.hour.fillna(0)
    pairs_df['shift_day_of_week'] = pairs_df['shift_start_date'].dt.dayofweek.fillna(0)
    pairs_df['shift_duration_hours'] = ((pairs_df['shift_end_date'] - pairs_df['shift_start_date']).dt.total_seconds() / 3600).fillna(8)
    pairs_df['is_night_shift'] = ((pairs_df['shift_hour'] >= 20) | (pairs_df['shift_hour'] < 6)).astype(int)
    pairs_df['lead_time_days'] = (pairs_df['shift_start_date'] - pd.Timestamp('today')).dt.days.fillna(1)
    pairs_df['experience_gap'] = pd.to_numeric(pairs_df['total_shifts_completed'], errors='coerce').fillna(0) - pd.to_numeric(pairs_df['experience_required'], errors='coerce').fillna(0)
    pairs_df['preferred_location_distance_rank'] = 1

    hospital_ids = shifts_df['hospital_id'].dropna().unique().tolist()
    for hid in hospital_ids:
        col_name = f"hospital_{hid}_familiarity"
        if col_name not in pairs_df.columns:
            pairs_df[col_name] = 0

    return pairs_df, hospital_ids