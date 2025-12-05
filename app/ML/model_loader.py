from pathlib import Path 
import logging
import joblib
import lightgbm as lgb
from typing import Optional,Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL_PATH_TXT = Path(__file__).parent / "models" / "nurse_shift_model.txt"
DEFAULT_MODEL_PATH_PICKLE = Path(__file__).parent / "models" / "nurse_shift_model.pkl"

_model: Optional[Any] = None


def load_model(path: Optional[Path]= None):
    path = Path(path) if path else (DEFAULT_MODEL_PATH_TXT if DEFAULT_MODEL_PATH_TXT.exists() else DEFAULT_MODEL_PATH_PICKLE)
    if not path.exists():
        logger.error(f"Model file not found at path: {path}")
        raise FileNotFoundError(f"Model file not found at path: {path}")
    
    if path.suffix  == ('.txt','.model'):
        model = lgb.Booster(model_file=str(path))
        logger.info(f"Loaded LightGBM Booster from text file: %s" ,path)
    else:
        model = joblib.load(path)
        logger.info("Loaded model from pickle/joblib file: %s (type=%s)", path, type(model))
        return model
    
def init(model_path: Optional[Path] = None):
    global _model
    if _model is None:
        _model = load_model(model_path)
    return _model

def get_model() -> Any:
    if _model is None:
        raise ValueError("Model not loaded. Call init() first.")
    return _model

