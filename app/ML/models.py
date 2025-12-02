import lightgbm as lgb
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "LightGBM.pkl"

def load_model():
    return joblib.load(MODEL_PATH)