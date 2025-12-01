import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# -----------------------------
# 0️⃣ Example synthetic nurses and shifts
# -----------------------------
# NOTE: This is temporary example data for training the model.
# In production, replace this with real data from your nurse and shift tables.
nurses_df = pd.DataFrame({
    'nurse_id': [f'n{i}' for i in range(1,6)],
    'specialization': ['Läkare','Sjuksköterska','Sjuksköterska','Arbetsterapeut','Läkare'],
    'base_location_norm': ['Söderstad','Söderstad','Nysås','Nysås','Söderstad'],
    'role_id': ['R1','R2','R2','R3','R1'],
    'date_of_birth': ['1990-01-01','1992-03-12','1988-07-23','1995-11-05','1990-01-01']
})

shifts_df = pd.DataFrame({
    'unique_shift_id': [f's{i}' for i in range(1,6)],
    'specialization_norm': ['Läkare','Sjuksköterska','Sjuksköterska','Arbetsterapeut','Läkare'],
    'location_norm': ['Söderstad','Söderstad','Nysås','Nysås','Söderstad'],
    'shift_start_date': pd.date_range('2025-12-01 08:00', periods=5, freq='h'),
    'shift_end_date': pd.date_range('2025-12-01 16:00', periods=5, freq='h'),
    'shift_type': ['DAY','NIGHT','DAY','DAY','NIGHT']
})

# -----------------------------
# 1️⃣ Create all possible nurse-shift pairs
# -----------------------------
pairs = pd.merge(
    nurses_df.assign(key=1),
    shifts_df.assign(key=1),
    on='key'
).drop('key', axis=1)

# -----------------------------
# 2️⃣ Create labels (temporary)
# -----------------------------
# Currently using simple rules to define matches.
# FUTURE: Replace this with historical application/completed shift data.
pairs['spec_match'] = (pairs['specialization'] == pairs['specialization_norm']).astype(int)
pairs['location_match'] = (pairs['base_location_norm'] == pairs['location_norm']).astype(int)
pairs['role_match'] = (pairs['role_id'] == 'R1').astype(int)  # TEMP: placeholder logic
pairs['label'] = ((pairs['spec_match']==1) & (pairs['location_match']==1) & (pairs['role_match']==1)).astype(int)

# -----------------------------
# Balance dataset: max 3x negatives per positive
# -----------------------------
positives = pairs[pairs['label']==1]
neg_count = min(len(pairs[pairs['label']==0]), len(positives)*3)
negatives = pairs[pairs['label']==0].sample(n=neg_count, random_state=42)
pairs_balanced = pd.concat([positives, negatives]).reset_index(drop=True)

# -----------------------------
# 3️⃣ Feature engineering
# -----------------------------
# NOTE: Some features are temporary/fake for training purposes.
pairs_balanced['shift_duration_h'] = (pairs_balanced['shift_end_date'] - pairs_balanced['shift_start_date']).dt.total_seconds() / 3600
pairs_balanced['shift_duration_h_bucket'] = pd.cut(
    pairs_balanced['shift_duration_h'],
    bins=[0,48,72,np.inf],
    labels=['short','medium','long']
)
pairs_balanced['shift_hour'] = pairs_balanced['shift_start_date'].dt.hour
pairs_balanced['shift_dayofweek'] = pairs_balanced['shift_start_date'].dt.dayofweek
pairs_balanced['is_night_shift'] = (pairs_balanced['shift_type']=='NIGHT').astype(int)
pairs_balanced['age'] = (pd.Timestamp('2025-12-01') - pd.to_datetime(pairs_balanced['date_of_birth'])).dt.days // 365
pairs_balanced['distance_km'] = np.random.randint(1,50,size=len(pairs_balanced))  # TEMP: replace with real distance calculation using lat/lng

# One-hot encode categorical features
cat_features = ['shift_type','shift_duration_h_bucket']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_features = ohe.fit_transform(pairs_balanced[cat_features])
ohe_df = pd.DataFrame(ohe_features, columns=ohe.get_feature_names_out(cat_features))

# Numerical features
num_features = ['shift_duration_h','spec_match','location_match','role_match','shift_hour','shift_dayofweek','is_night_shift','age','distance_km']
num_df = pairs_balanced[num_features].fillna(0)

X = pd.concat([num_df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
y = pairs_balanced['label']

# Standardize numerical features
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# -----------------------------
# 4️⃣ Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# -----------------------------
# 5️⃣ Train Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# -----------------------------
# 6️⃣ Evaluate
# -----------------------------
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"AUC: {auc:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
print("Label counts:\n", y.value_counts())

# -----------------------------
# 7️⃣ Save model, scaler, and encoder
# -----------------------------
joblib.dump(model, "nurse_shift_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(ohe, "ohe.pkl")

# -----------------------------
# 8️⃣ Function to score multiple shifts for a nurse
# -----------------------------
def score_shifts(nurse, shifts_df):
    """
    nurse: dict with keys 'specialization', 'base_location_norm', 'role_id', 'date_of_birth'
    shifts_df: DataFrame with same structure as shifts_df above
    
    returns: DataFrame with unique_shift_id + match_score (0–1)
    """
    temp_df = shifts_df.copy()
    
    # Pair features
    temp_df['spec_match'] = (nurse['specialization'] == temp_df['specialization_norm']).astype(int)
    temp_df['location_match'] = (nurse['base_location_norm'] == temp_df['location_norm']).astype(int)
    temp_df['role_match'] = int(nurse['role_id'] == 'R1')  # TEMP placeholder logic
    
    # Shift features
    temp_df['shift_duration_h'] = (temp_df['shift_end_date'] - temp_df['shift_start_date']).dt.total_seconds() / 3600
    temp_df['shift_duration_h_bucket'] = pd.cut(
        temp_df['shift_duration_h'],
        bins=[0,48,72,np.inf],
        labels=['short','medium','long']
    )
    temp_df['shift_hour'] = temp_df['shift_start_date'].dt.hour
    temp_df['shift_dayofweek'] = temp_df['shift_start_date'].dt.dayofweek
    temp_df['is_night_shift'] = (temp_df['shift_type']=='NIGHT').astype(int)
    
    # Temporary nurse features
    temp_df['age'] = (pd.Timestamp('2025-12-01') - pd.to_datetime(nurse['date_of_birth'])).days // 365
    temp_df['distance_km'] = np.random.randint(1,50,size=len(temp_df))  # TEMP: replace with real distance
    
    # One-hot encode categorical
    ohe_loaded = joblib.load("ohe.pkl")
    ohe_features = ohe_loaded.transform(temp_df[cat_features])
    ohe_df = pd.DataFrame(ohe_features, columns=ohe_loaded.get_feature_names_out(cat_features))
    
    # Numerical features
    num_df = temp_df[num_features].fillna(0)
    scaler_loaded = joblib.load("scaler.pkl")
    num_df[num_features] = scaler_loaded.transform(num_df[num_features])
    
    # Combine features
    X_input = pd.concat([num_df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
    
    # Prediction
    model_loaded = joblib.load("nurse_shift_model.pkl")
    temp_df['match_score'] = model_loaded.predict_proba(X_input)[:,1]
    
    return temp_df[['unique_shift_id','match_score']]

# ✅ Example usage
nurse_example = {'specialization':'Läkare','base_location_norm':'Söderstad','role_id':'R1','date_of_birth':'1990-01-01'}
print(score_shifts(nurse_example, shifts_df))