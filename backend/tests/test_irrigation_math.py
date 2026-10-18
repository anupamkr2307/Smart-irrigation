import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.water_service import (
    calculate_water_balance,
    calculate_penman_et0,
    calculate_usda_effective_rain,
    convert_area_to_sq_meters
)

def test_area_conversions():
    # 1 acre ~ 4046.86 m2
    sqm_1_acre = convert_area_to_sq_meters(1.0, "acres")
    assert 4046.0 <= sqm_1_acre <= 4047.0

    # 1 hectare = 10000 m2
    sqm_1_ha = convert_area_to_sq_meters(1.0, "hectares")
    assert sqm_1_ha == 10000.0

def test_penman_monteith_et0():
    # Moderate summer day in India: 30C, 50% humidity, 2 m/s wind
    et0 = calculate_penman_et0(temp_mean=30.0, humidity=50.0, wind_speed=2.0, solar_rad=18.0)
    assert 2.0 <= et0 <= 8.5, f"Expected realistic ET0, got {et0}"

def test_effective_rainfall():
    # Zero rainfall should give 0 effective rain
    assert calculate_usda_effective_rain(0.0) == 0.0

    # Moderate rain: 50 mm
    pe_50 = calculate_usda_effective_rain(50.0)
    assert 0 < pe_50 <= 50.0

    # High rainfall: 100 mm (runoff occurs, effective rain is substantially lower than total)
    pe_100 = calculate_usda_effective_rain(100.0)
    assert pe_100 < 60.0

def test_water_balance_nonnegative():
    # Case with massive rainfall (100 mm) and high soil moisture
    wb = calculate_water_balance(
        crop_kc=1.0,
        et0_mm=4.0,
        forecast_rain_mm=100.0,
        soil_type="loamy",
        soil_moisture_pct=28.0,
        farm_area=2.0,
        area_unit="acres",
        irrigation_method="Drip"
    )
    # Net irrigation MUST NOT be negative
    assert wb["net_irrigation_mm"] >= 0.0
    assert wb["net_irrigation_liters"] >= 0.0
    assert wb["gross_irrigation_liters"] >= 0.0

def test_water_balance_volume_consistency():
    # 1 mm depth across 1 m2 area = exactly 1 liter
    wb = calculate_water_balance(
        crop_kc=1.15,
        et0_mm=5.0,
        forecast_rain_mm=0.0,
        soil_type="sandy loam",
        soil_moisture_pct=10.0,
        farm_area=1000.0,
        area_unit="sqm",
        irrigation_method="Drip"
    )
    assert wb["net_irrigation_mm"] > 0
    # Volume in liters should match depth (mm) * area (1000 sqm)
    expected_liters = wb["net_irrigation_mm"] * 1000.0
    assert abs(wb["net_irrigation_liters"] - expected_liters) < 1.0

def test_gross_irrigation_efficiency():
    wb_drip = calculate_water_balance(
        crop_kc=1.0, et0_mm=5.0, forecast_rain_mm=0.0, soil_type="loamy",
        soil_moisture_pct=15.0, farm_area=1.0, area_unit="acres", irrigation_method="Drip"
    )
    wb_furrow = calculate_water_balance(
        crop_kc=1.0, et0_mm=5.0, forecast_rain_mm=0.0, soil_type="loamy",
        soil_moisture_pct=15.0, farm_area=1.0, area_unit="acres", irrigation_method="Furrow"
    )
    # Drip (90% eff) should require less gross water than furrow (60% eff) for identical net requirement
    assert wb_drip["gross_irrigation_mm"] < wb_furrow["gross_irrigation_mm"]
