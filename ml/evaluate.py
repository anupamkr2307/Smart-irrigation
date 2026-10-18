"""
Model Diagnostics and Evaluation Inspector
Loads the trained model and runs diagnostics across crop stages and soil types.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

def run_diagnostics():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    model_path = os.path.join(models_dir, "irrigation_model.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")

    if not os.path.exists(model_path) or not os.path.exists(metrics_path):
        raise FileNotFoundError("Model artifacts not found. Run train.py first.")

    pipeline = joblib.load(model_path)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    print("Loaded Model Metrics:")
    print(json.dumps(metrics, indent=2))

    # Test sample inference
    sample_input = pd.DataFrame([{
        "soil_type": "loamy",
        "crop_type": "Wheat",
        "crop_growth_stage": "Mid-Season",
        "temperature_c": 24.5,
        "relative_humidity_pct": 45.0,
        "soil_moisture_pct": 22.0,
        "recent_rainfall_mm": 5.0,
        "forecast_rainfall_mm": 2.0,
        "wind_speed_ms": 2.5,
        "solar_radiation_mj": 18.0,
        "evapotranspiration_et0_mm": 4.8
    }])

    pred = float(pipeline.predict(sample_input)[0])
    pred_clipped = max(0.0, pred)
    print(f"\nSample Prediction for Wheat (Mid-Season, Loamy, 24.5C): {pred_clipped:.2f} mm")

if __name__ == "__main__":
    run_diagnostics()
