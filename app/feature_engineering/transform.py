# transform.py
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
from geopy.distance import geodesic

# ------------------------------
# Load data
# ------------------------------
def load_data():
    engine = create_engine("postgresql://localhost:5432/phdb?user=postgres&password=password")
    shifts = pd.read_sql("SELECT * FROM shift_advertisement", engine)
    nurses = pd.read_sql("SELECT * FROM nurse", engine)
    nurse_shifts = pd.read_sql("SELECT * FROM nurse_shift", engine)
    return nurses, shifts, nurse_shifts

# ------------------------------
# Feature Engineering
# ------------------------------
def transform(nurses: pd.DataFrame, shifts: pd.DataFrame, nurse_shifts: pd.DataFrame):
    # Merge nurses och shifts via nurse_shifts
    df = nurse_shifts.merge(nurses, left_on="nurse_id", right_on="id", suffixes=("", "_nurse"))
    df = df.merge(shifts, left_on="shift_id", right_on="id", suffixes=("", "_shift"))

    # ------------------------------
    # Time-based features
    # ------------------------------
    df['shift_start_date'] = pd.to_datetime(df['shift_start_date'])
    df['shift_end_date'] = pd.to_datetime(df['shift_end_date'])
    df['duration_hours'] = (df['shift_end_date'] - df['shift_start_date']).dt.total_seconds() / 3600

    # ------------------------------
    # Specialization / role match
    # ------------------------------
    df['specialization_match'] = (df['specialization'] == df['specialization_shift']).astype(int)
    
    if 'role_id' in df.columns and 'role_id_shift' in df.columns:
        df['role_match'] = (df['role_id'] == df['role_id_shift']).astype(int)
    else:
        df['role_match'] = 1  # default match

    # ------------------------------
    # Experience gap
    # ------------------------------
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df['nurse_age'] = (pd.Timestamp.now() - df['date_of_birth']).dt.days / 365.25
    df['experience_gap'] = df['nurse_age'] - df['duration_hours'] / 2080  # assuming 2080 work hours/year

    # ------------------------------
    # Location distance
    # ------------------------------
    def calc_distance(row):
        try:
            nurse_loc = (float(row['base_location_lat']), float(row['base_location_lng']))
            shift_loc = (float(row['lat_shift']), float(row['lng_shift']))
            return geodesic(nurse_loc, shift_loc).km
        except:
            return np.nan

    if 'base_location_lat' in df.columns and 'lat_shift' in df.columns:
        df['distance_km'] = df.apply(calc_distance, axis=1)
    else:
        df['distance_km'] = np.nan

    # ------------------------------
    # Categorical encoding
    # ------------------------------
    categorical_cols = ['specialization', 'specialization_shift', 'shift_type']
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

    # ------------------------------
    # Numerical features
    # ------------------------------
    numerical_cols = ['duration_hours', 'nurse_age', 'experience_gap', 'distance_km', 'specialization_match', 'role_match']
    df[numerical_cols] = df[numerical_cols].fillna(0)

    # ------------------------------
    # Save encoder for future inference
    # ------------------------------
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    # ------------------------------
    # Return combined feature matrix
    # ------------------------------
    return pd.concat([df[numerical_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# ------------------------------
# Real-time inference helper
# ------------------------------
def transform_single(nurse_row: dict, shift_row: dict, encoder_path='encoder.pkl'):
    """
    Transform a single nurse + shift pair for inference
    """
    df = pd.DataFrame([ {**nurse_row, **shift_row} ])
    # duplicate code for features, but without fitting encoder
    df['shift_start_date'] = pd.to_datetime(df['shift_start_date'])
    df['shift_end_date'] = pd.to_datetime(df['shift_end_date'])
    df['duration_hours'] = (df['shift_end_date'] - df['shift_start_date']).dt.total_seconds() / 3600

    df['specialization_match'] = (df['specialization'] == df['specialization_shift']).astype(int)
    df['role_match'] = 1
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df['nurse_age'] = (pd.Timestamp.now() - df['date_of_birth']).dt.days / 365.25
    df['experience_gap'] = df['nurse_age'] - df['duration_hours'] / 2080

    try:
        nurse_loc = (float(df.loc[0,'base_location_lat']), float(df.loc[0,'base_location_lng']))
        shift_loc = (float(df.loc[0,'lat_shift']), float(df.loc[0,'lng_shift']))
        df['distance_km'] = geodesic(nurse_loc, shift_loc).km
    except:
        df['distance_km'] = 0

    categorical_cols = ['specialization', 'specialization_shift', 'shift_type']
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')

    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    encoded_features = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

    numerical_cols = ['duration_hours', 'nurse_age', 'experience_gap', 'distance_km', 'specialization_match', 'role_match']
    df[numerical_cols] = df[numerical_cols].fillna(0)

    return pd.concat([df[numerical_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# ------------------------------
# Run script
# ------------------------------
if __name__ == "__main__":
    nurses, shifts, nurse_shifts = load_data()
    features = transform(nurses, shifts, nurse_shifts)
    print("Feature matrix shape:", features.shape)
    display(features.head())