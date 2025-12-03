import lightgbm as lgb
import joblib
from pathlib import Path

# This module loads the trained LightGBM model used for prototyping.
# This prototype is not meant for production use as is.
# 
# Why LightGBM?
# - I tested multiple algorithms (e.g., RandomForest, GradientBoosting, XGBoost, etc.)
#   and LightGBM performed best on the synthetic/tabular data used in this prototype.
# - It is fast, efficient and handles tabular data very well.
#
# IMPORTANT:
# When switching to real production data:
# 1. Retrain the model entirely using real historical data.
# 2. Benchmark different ML algorithms again – the best model may change with real data.
# 3. Perform proper hyperparameter tuning to maximize performance.
# 4. Select the model that performs best on real evaluation metrics.
#
# This file only contains the logic for loading the trained model. 
# If another algorithm is used in the future, replace the loading mechanism here.

MODEL_PATH = Path(__file__).parent / "models" / "LightGBM.pkl"

def load_model():
    return joblib.load(MODEL_PATH)