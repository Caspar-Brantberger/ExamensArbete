import pandas as pd
import numpy as np
from app.ML.models import model, scaler, ohe

# -----------------------------
# 1️⃣ Define which features are used
# -----------------------------
cat_features = ['shift_type', 'shift_duration_h_bucket']
num_features = ['shift_duration_h', 'spec_match', 'location_match', 'role_match',
                'shift_hour', 'shift_dayofweek', 'is_night_shift', 'age', 'distance_km']

# -----------------------------
# 2️⃣ Function to score multiple shifts
# -----------------------------
def score_shifts(nurse, shifts_df):
    """
    nurse: dict with keys like 'specialization', 'base_location_norm', 'role_id', 'date_of_birth'
    shifts_df: DataFrame containing shift data
    
    return: DataFrame with 'unique_shift_id' + 'match_score' (0–1)
    """
    temp_df = shifts_df.copy()
    
    # Pair features
    temp_df['spec_match'] = (nurse['specialization'] == temp_df['specialization_norm']).astype(int)
    temp_df['location_match'] = (nurse['base_location_norm'] == temp_df['location_norm']).astype(int)
    temp_df['role_match'] = int(nurse['role_id'] == 'R1')  # Temporary logic
    
    # Shift features
    temp_df['shift_duration_h'] = (temp_df['shift_end_date'] - temp_df['shift_start_date']).dt.total_seconds() / 3600
    temp_df['shift_duration_h_bucket'] = pd.cut(temp_df['shift_duration_h'],
                                                bins=[0,48,72,np.inf],
                                                labels=['short','medium','long'])
    temp_df['shift_hour'] = temp_df['shift_start_date'].dt.hour
    temp_df['shift_dayofweek'] = temp_df['shift_start_date'].dt.dayofweek
    temp_df['is_night_shift'] = (temp_df['shift_type']=='NIGHT').astype(int)
    
    # Temporary nurse features
    temp_df['age'] = (pd.Timestamp('2025-12-01') - pd.to_datetime(nurse['date_of_birth'])).days // 365
    temp_df['distance_km'] = np.random.randint(1,50,size=len(temp_df))  # Temporary
    
    # One-hot encode categorical features
    ohe_features = ohe.transform(temp_df[cat_features])
    ohe_df = pd.DataFrame(ohe_features, columns=ohe.get_feature_names_out(cat_features))
    
    # Scale numeric features
    num_df = temp_df[num_features].fillna(0)
    num_df[num_features] = scaler.transform(num_df[num_features])
    
    # Combine features
    X_input = pd.concat([num_df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
    
    # Predict match scores
    temp_df['match_score'] = model.predict_proba(X_input)[:,1]
    
    return temp_df[['unique_shift_id', 'match_score']]

# -----------------------------
# 3️⃣ Example usage
# -----------------------------
if __name__ == "__main__":
    nurse_example = {
        'specialization':'Läkare',
        'base_location_norm':'Söderstad',
        'role_id':'R1',
        'date_of_birth':'1990-01-01'
    }
    
    example_shifts = pd.DataFrame({
        'unique_shift_id': [f's{i}' for i in range(1,6)],
        'specialization_norm': ['Läkare','Sjuksköterska','Sjuksköterska','Arbetsterapeut','Läkare'],
        'location_norm': ['Söderstad','Söderstad','Nysås','Nysås','Söderstad'],
        'shift_start_date': pd.date_range('2025-12-01 08:00', periods=5, freq='h'),
        'shift_end_date': pd.date_range('2025-12-01 16:00', periods=5, freq='h'),
        'shift_type': ['DAY','NIGHT','DAY','DAY','NIGHT']
    })
    
    print(score_shifts(nurse_example, example_shifts))