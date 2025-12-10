# test_transform.py
import pytest
import pandas as pd
import numpy as np
import os
from app.feature_engineering.transform import transform, load_data, transform_single



# ------------------------------
# Mock data fixtures
# ------------------------------
@pytest.fixture
def nurses_df():
    return pd.DataFrame({
        'id': [1],
        'specialization': ['ICU'],
        'role_id': [101],
        'date_of_birth': ['1990-01-01'],
        'base_location_lat': [59.3293],
        'base_location_lng': [18.0686]
    })

@pytest.fixture
def shifts_df():
    return pd.DataFrame({
        'id': [10],
        'specialization_shift': ['ICU'],
        'role_id_shift': [101],
        'shift_start_date': ['2025-12-10 08:00:00'],
        'shift_end_date': ['2025-12-10 16:00:00'],
        'shift_type': ['Day'],
        'lat_shift': [59.332],
        'lng_shift': [18.065]
    })

@pytest.fixture
def nurse_shifts_df():
    return pd.DataFrame({
        'nurse_id': [1],
        'shift_id': [10]
    })

# ------------------------------
# Unit tests
# ------------------------------
def test_transform_shape(nurses_df, shifts_df, nurse_shifts_df):
    """Test that transform returns correct columns and rows"""
    features = transform(nurses_df, shifts_df, nurse_shifts_df)
    expected_cols = ['duration_hours', 'nurse_age', 'experience_gap', 'distance_km', 'specialization_match', 'role_match']
    for col in expected_cols:
        assert col in features.columns
    assert features.shape[0] == 1  

def test_transform_single_creates_correct_features(nurses_df, shifts_df):
    """Test transform_single produces same columns as transform"""
    nurse_row = nurses_df.iloc[0].to_dict()
    shift_row = shifts_df.iloc[0].to_dict()

    # Ensure encoder.pkl exists
    transform(nurses_df, shifts_df, pd.DataFrame({'nurse_id':[1],'shift_id':[10]}))
    
    features_single = transform_single(nurse_row, shift_row)
    numerical_cols = ['duration_hours', 'nurse_age', 'experience_gap', 'distance_km', 'specialization_match', 'role_match']
    for col in numerical_cols:
        assert col in features_single.columns
    assert features_single.shape[0] == 1

def test_distance_calculation(nurses_df, shifts_df, nurse_shifts_df):
    """Test that distance_km is positive"""
    features = transform(nurses_df, shifts_df, nurse_shifts_df)
    assert features['distance_km'].iloc[0] > 0

def test_specialization_role_match(nurses_df, shifts_df, nurse_shifts_df):
    """Test specialization_match and role_match are correct for exact matches"""
    features = transform(nurses_df, shifts_df, nurse_shifts_df)
    assert features['specialization_match'].iloc[0] == 1
    assert features['role_match'].iloc[0] == 1

# ------------------------------
# Cleanup
# ------------------------------
def teardown_module(module):
    """Remove encoder.pkl after tests"""
    if os.path.exists('encoder.pkl'):
        os.remove('encoder.pkl')
