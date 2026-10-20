"""
ML Prediction Service
Loads the lightweight model weights or serialized Pipeline, executes inference,
and returns feature contributions and confidence intervals with zero C-dependency overhead.
"""

import os
import json

try:
    import joblib
except ImportError:
    joblib = None

class MLService:
    def __init__(self, model_path=None, metrics_path=None, weights_path=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        self.weights_path = weights_path or os.path.join(base_dir, "ml", "models", "model_weights.json")
        self.model_path = model_path or os.path.join(base_dir, "ml", "models", "irrigation_model.joblib")
        self.metrics_path = metrics_path or os.path.join(base_dir, "ml", "models", "metrics.json")
        self.pipeline = None
        self.model_weights = None
        self.metrics = {}
        self.load_model()

    def load_model(self):
        # 1. Try lightweight model weights first (zero-dependency, instant on serverless)
        if os.path.exists(self.weights_path):
            try:
                with open(self.weights_path, "r", encoding="utf-8") as f:
                    self.model_weights = json.load(f)
                    self.pipeline = self.model_weights  # Sentinel for pipeline-loaded checks
            except Exception as e:
                print(f"Error loading model weights from {self.weights_path}: {e}")

        # 2. Fallback to joblib pipeline if available
        if self.model_weights is None and os.path.exists(self.model_path) and joblib is not None:
            try:
                self.pipeline = joblib.load(self.model_path)
            except Exception as e:
                print(f"Error loading model from {self.model_path}: {e}")

        # 3. Load metrics metadata
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except Exception as e:
                print(f"Error loading metrics from {self.metrics_path}: {e}")
                self.metrics = {}

    def predict(self, feature_dict):
        """
        Runs ML prediction for net irrigation requirement in mm.
        Supports both zero-dependency pure Python linear evaluation and scikit-learn Pipeline.
        """
        if self.pipeline is None and self.model_weights is None:
            self.load_model()
            if self.pipeline is None and self.model_weights is None:
                raise RuntimeError("ML model pipeline is not loaded.")

        if self.model_weights is not None:
            # Pure Python Linear Regression execution (0 runtime dependencies)
            pred = self.model_weights.get("intercept", 4.0315)
            cat_w = self.model_weights.get("categorical_weights", {})

            soil = feature_dict.get("soil_type")
            crop = feature_dict.get("crop_type")
            stage = feature_dict.get("crop_growth_stage")

            if soil:
                pred += cat_w.get(f"soil_type_{soil}", 0.0)
            if crop:
                pred += cat_w.get(f"crop_type_{crop}", 0.0)
            if stage:
                pred += cat_w.get(f"crop_growth_stage_{stage}", 0.0)

            cols = self.model_weights.get("numerical_cols", [])
            means = self.model_weights.get("num_means", [])
            scales = self.model_weights.get("num_scales", [])
            weights = self.model_weights.get("num_weights", [])

            for col, m, s, w in zip(cols, means, scales, weights):
                val = float(feature_dict.get(col, m))
                z = (val - m) / s if s != 0 else 0.0
                pred += z * w

            raw_pred = float(pred)
        else:
            # Fallback if joblib pipeline loaded
            import pandas as pd
            df_input = pd.DataFrame([feature_dict])
            raw_pred = float(self.pipeline.predict(df_input)[0])
        # Physical lower bound: irrigation requirement cannot be negative
        predicted_mm = round(max(0.0, raw_pred), 2)

        # Confidence and uncertainty based on test MAE
        mae = self.metrics.get("metrics", {}).get("mae", 1.78)
        lower_bound = round(max(0.0, predicted_mm - mae), 2)
        upper_bound = round(predicted_mm + mae, 2)

        # Factor contributions for transparency
        top_factors = []
        if "top_feature_weights" in self.metrics:
            crop_name = feature_dict.get("crop_type", "")
            stage = feature_dict.get("crop_growth_stage", "")
            for item in self.metrics["top_feature_weights"][:6]:
                f_name = item["feature"]
                weight = item["weight"]
                if "et0" in f_name:
                    top_factors.append(f"High atmospheric evaporative demand ({feature_dict.get('evapotranspiration_et0_mm')} mm/day) increases water demand.")
                elif "forecast_rainfall" in f_name and feature_dict.get("forecast_rainfall_mm", 0) > 0:
                    top_factors.append(f"Upcoming rainfall ({feature_dict.get('forecast_rainfall_mm')} mm) directly offsets net irrigation need.")
                elif "soil_moisture" in f_name:
                    top_factors.append(f"Current soil moisture ({feature_dict.get('soil_moisture_pct')}%) reserves buffer plant stress.")
                elif crop_name and crop_name in f_name:
                    direction = "increases" if weight > 0 else "moderates"
                    top_factors.append(f"Crop physiological coefficient for {crop_name} {direction} required depth.")
                elif stage and stage in f_name:
                    top_factors.append(f"Growth stage ({stage}) factor applied.")

        if not top_factors:
            top_factors = [
                "Evapotranspiration rate (ET0) drives primary crop water need.",
                "Upcoming forecast rainfall reduces net supplementary water required.",
                "Current root zone soil moisture provides an immediate buffer."
            ]

        return {
            "predicted_net_mm": predicted_mm,
            "uncertainty_interval_mm": [lower_bound, upper_bound],
            "mae_confidence_mm": mae,
            "model_version": self.metrics.get("model_name", "LinearRegression Pipeline v1.0"),
            "r2_score": self.metrics.get("metrics", {}).get("r2_score", 0.70),
            "main_factors": top_factors[:4]
        }

# Global singleton
ml_service = MLService()
