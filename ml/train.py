"""
Machine Learning Training Pipeline
Trains a scikit-learn LinearRegression pipeline with OneHotEncoder and StandardScaler.
Evaluates MAE, RMSE, and R2 on held-out test split, and persists the model with Joblib.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import validate_and_preprocess

CATEGORICAL_FEATURES = ["soil_type", "crop_type", "crop_growth_stage"]
NUMERICAL_FEATURES = [
    "temperature_c",
    "relative_humidity_pct",
    "soil_moisture_pct",
    "recent_rainfall_mm",
    "forecast_rainfall_mm",
    "wind_speed_ms",
    "solar_radiation_mj",
    "evapotranspiration_et0_mm"
]
TARGET = "net_irrigation_req_mm"

def train_model():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "irrigation_data.csv"))
    df = validate_and_preprocess(dataset_path)

    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df[TARGET]

    # 80/20 train-test split with reproducible seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )

    # Preprocessing transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERICAL_FEATURES)
        ]
    )

    # Full scikit-learn Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = pipeline.predict(X_test)
    # Clip predictions to nonnegative since irrigation cannot be negative physically
    y_pred_clipped = np.clip(y_pred, 0.0, None)

    mae = float(mean_absolute_error(y_test, y_pred_clipped))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_clipped)))
    r2 = float(r2_score(y_test, y_pred_clipped))

    print("================ Model Evaluation Metrics ================")
    print(f"Mean Absolute Error (MAE): {mae:.4f} mm")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} mm")
    print(f"R-squared Score (R2): {r2:.4f}")
    print("==========================================================")

    # Extract feature names & coefficients for explainability
    ohe_columns = list(pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = ohe_columns + NUMERICAL_FEATURES
    coefficients = list(pipeline.named_steps["regressor"].coef_)
    intercept = float(pipeline.named_steps["regressor"].intercept_)

    importance = sorted(
        [{"feature": f, "weight": float(c)} for f, c in zip(feature_names, coefficients)],
        key=lambda x: abs(x["weight"]),
        reverse=True
    )

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    os.makedirs(models_dir, exist_ok=True)
    model_file = os.path.join(models_dir, "irrigation_model.joblib")
    metrics_file = os.path.join(models_dir, "metrics.json")
    weights_file = os.path.join(models_dir, "model_weights.json")

    joblib.dump(pipeline, model_file)
    print(f"Trained pipeline persisted to: {model_file}")

    # Export lightweight zero-dependency model weights for serverless deployments
    num_transformer = pipeline.named_steps["preprocessor"].named_transformers_["num"]
    weights_dict = {f: float(w) for f, w in zip(feature_names, coefficients)}
    model_spec = {
        "intercept": intercept,
        "categorical_weights": weights_dict,
        "numerical_cols": NUMERICAL_FEATURES,
        "num_means": [float(m) for m in num_transformer.mean_],
        "num_scales": [float(s) for s in num_transformer.scale_],
        "num_weights": [float(weights_dict[c]) for c in NUMERICAL_FEATURES]
    }
    with open(weights_file, "w", encoding="utf-8") as f:
        json.dump(model_spec, f, indent=2)
    print(f"Lightweight model weights saved to: {weights_file}")

    metadata = {
        "model_name": "LinearRegression Irrigation Pipeline",
        "algorithm": "Ordinary Least Squares Linear Regression",
        "evaluation_date": "2026-10-18",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4)
        },
        "intercept": round(intercept, 4),
        "top_feature_weights": importance[:10],
        "input_features": {
            "categorical": CATEGORICAL_FEATURES,
            "numerical": NUMERICAL_FEATURES
        }
    }

    with open(metrics_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")

    return metadata

if __name__ == "__main__":
    train_model()
