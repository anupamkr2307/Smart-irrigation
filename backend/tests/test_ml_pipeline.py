import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.ml_service import ml_service

def test_ml_model_loaded():
    assert ml_service.pipeline is not None
    assert "metrics" in ml_service.metrics

def test_ml_prediction_validity():
    inputs = {
        "soil_type": "loamy",
        "crop_type": "Wheat",
        "crop_growth_stage": "Mid-Season",
        "temperature_c": 26.0,
        "relative_humidity_pct": 45.0,
        "soil_moisture_pct": 20.0,
        "recent_rainfall_mm": 0.0,
        "forecast_rainfall_mm": 0.0,
        "wind_speed_ms": 2.0,
        "solar_radiation_mj": 17.0,
        "evapotranspiration_et0_mm": 4.5
    }
    result = ml_service.predict(inputs)
    assert "predicted_net_mm" in result
    assert result["predicted_net_mm"] >= 0.0
    assert len(result["uncertainty_interval_mm"]) == 2
    assert result["uncertainty_interval_mm"][0] <= result["predicted_net_mm"] <= result["uncertainty_interval_mm"][1]
    assert len(result["main_factors"]) > 0

def test_ml_prediction_rainfall_suppression():
    # Comparing zero forecast rain vs heavy forecast rain
    base_inputs = {
        "soil_type": "loamy",
        "crop_type": "Wheat",
        "crop_growth_stage": "Mid-Season",
        "temperature_c": 26.0,
        "relative_humidity_pct": 45.0,
        "soil_moisture_pct": 20.0,
        "recent_rainfall_mm": 0.0,
        "wind_speed_ms": 2.0,
        "solar_radiation_mj": 17.0,
        "evapotranspiration_et0_mm": 4.5
    }
    dry_case = ml_service.predict({**base_inputs, "forecast_rainfall_mm": 0.0})
    wet_case = ml_service.predict({**base_inputs, "forecast_rainfall_mm": 40.0})

    # Expected: wet case should require less or equal net irrigation than dry case
    assert wet_case["predicted_net_mm"] <= dry_case["predicted_net_mm"]
