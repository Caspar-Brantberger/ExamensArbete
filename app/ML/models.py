import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "Models"

model = joblib.load(BASE_DIR / "nurse_shift_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
ohe = joblib.load(BASE_DIR / "ohe.pkl")