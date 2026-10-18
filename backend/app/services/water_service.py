"""
Agricultural Water Engine
Transparent implementation of FAO-56 Penman-Monteith, USDA-SCS effective rainfall,
soil water balance, and unit conversions.
"""

import numpy as np

# Soil water holding constants
SOIL_WATER_PROPERTIES = {
    "sandy": {
        "wilting_point_pct": 6.0,
        "field_capacity_pct": 14.0,
        "available_water_capacity_mm_per_m": 60.0,
        "infiltration_rate_mm_hr": 25.0,
        "recommended_method": "Drip",
        "precautions": "High permeability and low water retention. Avoid flood/basin irrigation to prevent deep leaching of nutrients. Apply frequent, small water pulses."
    },
    "loamy": {
        "wilting_point_pct": 12.0,
        "field_capacity_pct": 28.0,
        "available_water_capacity_mm_per_m": 140.0,
        "infiltration_rate_mm_hr": 12.0,
        "recommended_method": "Drip / Sprinkler",
        "precautions": "Ideal soil structure with balanced aeration and retention. Suitable for both drip and micro-sprinkler systems."
    },
    "clay": {
        "wilting_point_pct": 20.0,
        "field_capacity_pct": 38.0,
        "available_water_capacity_mm_per_m": 180.0,
        "infiltration_rate_mm_hr": 4.0,
        "recommended_method": "Drip / Furrow",
        "precautions": "Slow infiltration and high risk of waterlogging. Space irrigation intervals wider and monitor drainage to prevent root hypoxia."
    },
    "silty": {
        "wilting_point_pct": 14.0,
        "field_capacity_pct": 30.0,
        "available_water_capacity_mm_per_m": 160.0,
        "infiltration_rate_mm_hr": 8.0,
        "recommended_method": "Drip / Sprinkler",
        "precautions": "Prone to surface crusting if watered by overhead impact. Drip or gentle low-pressure sprinklers recommended."
    },
    "sandy loam": {
        "wilting_point_pct": 8.0,
        "field_capacity_pct": 20.0,
        "available_water_capacity_mm_per_m": 100.0,
        "infiltration_rate_mm_hr": 18.0,
        "recommended_method": "Drip / Sprinkler",
        "precautions": "Moderate water holding. Drip irrigation prevents excessive percolation below root zone."
    },
    "clay loam": {
        "wilting_point_pct": 16.0,
        "field_capacity_pct": 32.0,
        "available_water_capacity_mm_per_m": 160.0,
        "infiltration_rate_mm_hr": 7.0,
        "recommended_method": "Drip / Furrow",
        "precautions": "Good moisture retention. Ensure soil is allowed to dry to 50% available water before subsequent irrigation."
    }
}

IRRIGATION_EFFICIENCY = {
    "Drip": 0.90,       # 90% efficient
    "Sprinkler": 0.75,  # 75% efficient
    "Furrow": 0.60,     # 60% efficient
    "Basin": 0.50       # 50% efficient
}

def convert_area_to_sq_meters(area_val, unit="acres"):
    unit = unit.lower().strip()
    if unit in ["acre", "acres"]:
        return float(area_val) * 4046.85642
    elif unit in ["hectare", "hectares", "ha"]:
        return float(area_val) * 10000.0
    elif unit in ["sqm", "m2", "sq_meters"]:
        return float(area_val)
    return float(area_val) * 4046.86

def calculate_penman_et0(temp_mean, humidity, wind_speed=2.0, solar_rad=15.0):
    """FAO-56 reference evapotranspiration in mm/day."""
    es = 0.6108 * np.exp((17.27 * temp_mean) / (temp_mean + 237.3))
    ea = es * (humidity / 100.0)
    delta = (4098.0 * es) / ((temp_mean + 237.3) ** 2)
    gamma = 0.067
    rn = max(1.0, 0.408 * 0.6 * solar_rad)

    numerator = 0.408 * delta * rn + gamma * (900.0 / (temp_mean + 273.0)) * wind_speed * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * wind_speed)
    et0 = float(numerator / denominator)
    return max(1.5, min(10.0, et0))

def calculate_usda_effective_rain(forecast_rain_mm):
    """USDA Soil Conservation Service effective rainfall formula."""
    p = max(0.0, float(forecast_rain_mm))
    if p <= 0:
        return 0.0
    elif p <= 75.0:
        pe = p * (125.0 - 0.2 * p) / 125.0
    else:
        pe = (125.0 / 3.0) + (0.1 * p)
    return round(min(pe, p), 2)

def calculate_water_balance(
    crop_kc,
    et0_mm,
    forecast_rain_mm,
    soil_type,
    soil_moisture_pct,
    farm_area,
    area_unit="acres",
    irrigation_method="Drip",
    cycle_days=3
):
    """
    Computes complete agronomic water balance:
    - Crop Water Requirement (ETc)
    - Effective Rainfall (Pe)
    - Usable Soil Moisture Contribution
    - Net Irrigation (mm & Liters)
    - Gross Irrigation accounting for efficiency
    """
    soil_key = soil_type.lower().strip()
    soil_props = SOIL_WATER_PROPERTIES.get(soil_key, SOIL_WATER_PROPERTIES["loamy"])

    # 1. ETc over agricultural cycle
    daily_etc = et0_mm * crop_kc
    total_etc = daily_etc * cycle_days

    # 2. Effective Rainfall
    pe = calculate_usda_effective_rain(forecast_rain_mm)

    # 3. Available Soil Moisture Contribution
    wp = soil_props["wilting_point_pct"]
    fc = soil_props["field_capacity_pct"]
    # If soil moisture is above wilting point, crop can use part of it before stress
    usable_sm_depth_mm = max(0.0, (soil_moisture_pct - wp) * 0.1 * 6.0) # 60 cm root zone assumption

    # 4. Net Irrigation Requirement
    net_irrigation_mm = max(0.0, total_etc - pe - (usable_sm_depth_mm * 0.4))
    net_irrigation_mm = round(net_irrigation_mm, 2)

    # 5. Gross Irrigation Requirement
    efficiency = IRRIGATION_EFFICIENCY.get(irrigation_method, 0.90)
    gross_irrigation_mm = round(net_irrigation_mm / efficiency, 2) if net_irrigation_mm > 0 else 0.0

    # 6. Volume conversion: 1 mm depth * 1 m2 area = 1 Liter
    area_sqm = convert_area_to_sq_meters(farm_area, area_unit)
    net_liters = round(net_irrigation_mm * area_sqm, 1)
    gross_liters = round(gross_irrigation_mm * area_sqm, 1)
    rainfall_volume_liters = round(forecast_rain_mm * area_sqm, 1)
    effective_rain_liters = round(pe * area_sqm, 1)

    # 7. Irrigation schedule and duration
    if net_irrigation_mm == 0:
        schedule = "No irrigation required for the next 72 hours. Soil moisture and expected rainfall are adequate."
    elif net_irrigation_mm < 8.0:
        schedule = f"Light irrigation: 1 session of ~45 mins via {irrigation_method} over the next 48 hours."
    elif net_irrigation_mm < 20.0:
        schedule = f"Standard irrigation: 2 split applications of ~1.5 hours each over 3 days via {irrigation_method}."
    else:
        schedule = f"Peak demand irrigation: Daily application of 2-3 hours via {irrigation_method} to prevent moisture stress."

    # Water saving tips tailored to method
    water_saving_tips = [
        "Apply irrigation during early morning (5:00 AM - 8:00 AM) or late afternoon to minimize evaporation losses by up to 25%.",
        "Consider organic or plastic mulching around crop beds to retain soil moisture and reduce weeding.",
        f"For {soil_type} soil, maintain soil moisture within {wp:.1f}% to {fc:.1f}% to avoid plant stress and nutrient leaching."
    ]

    return {
        "daily_etc_mm": round(daily_etc, 2),
        "cycle_etc_mm": round(total_etc, 2),
        "effective_rainfall_mm": pe,
        "effective_rainfall_liters": effective_rain_liters,
        "forecast_rainfall_liters": rainfall_volume_liters,
        "usable_soil_water_mm": round(usable_sm_depth_mm, 2),
        "net_irrigation_mm": net_irrigation_mm,
        "net_irrigation_liters": net_liters,
        "gross_irrigation_mm": gross_irrigation_mm,
        "gross_irrigation_liters": gross_liters,
        "irrigation_efficiency_pct": round(efficiency * 100, 1),
        "schedule": schedule,
        "soil_precautions": soil_props["precautions"],
        "recommended_method": irrigation_method or soil_props["recommended_method"],
        "water_saving_tips": water_saving_tips,
        "area_sqm": round(area_sqm, 2)
    }
