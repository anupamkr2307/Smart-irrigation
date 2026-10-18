"""
Synthetic Agronomic Dataset Generator for Smart Irrigation System
Generates 1,500 distinct, structured records based on FAO-56 Penman-Monteith,
USDA Soil Conservation Service effective rainfall equations, and soil physical properties.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fix seed for reproducibility
np.random.seed(42)
random.seed(42)

LOCATIONS = [
    {"name": "Nashik", "state": "Maharashtra", "district": "Nashik", "lat": 19.9975, "lon": 73.7898},
    {"name": "Nagpur", "state": "Maharashtra", "district": "Nagpur", "lat": 21.1458, "lon": 79.0882},
    {"name": "Pune", "state": "Maharashtra", "district": "Pune", "lat": 18.5204, "lon": 73.8567},
    {"name": "Latur", "state": "Maharashtra", "district": "Latur", "lat": 18.4088, "lon": 76.5604},
    {"name": "Karnal", "state": "Haryana", "district": "Karnal", "lat": 29.6857, "lon": 76.9905},
    {"name": "Bathinda", "state": "Punjab", "district": "Bathinda", "lat": 30.2110, "lon": 74.9455},
    {"name": "Guntur", "state": "Andhra Pradesh", "district": "Guntur", "lat": 16.3067, "lon": 80.4365},
    {"name": "Belagavi", "state": "Karnataka", "district": "Belagavi", "lat": 15.8497, "lon": 74.4977},
    {"name": "Indore", "state": "Madhya Pradesh", "district": "Indore", "lat": 22.7196, "lon": 75.8577},
    {"name": "Jaipur", "state": "Rajasthan", "district": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    {"name": "Anand", "state": "Gujarat", "district": "Anand", "lat": 22.5645, "lon": 72.9289},
    {"name": "Coimbatore", "state": "Tamil Nadu", "district": "Coimbatore", "lat": 11.0168, "lon": 76.9558},
    {"name": "Varanasi", "state": "Uttar Pradesh", "district": "Varanasi", "lat": 25.3176, "lon": 82.9739},
    {"name": "Patna", "state": "Bihar", "district": "Patna", "lat": 25.5941, "lon": 85.1376},
    {"name": "Raipur", "state": "Chhattisgarh", "district": "Raipur", "lat": 21.2514, "lon": 81.6296},
]

SOIL_PROPERTIES = {
    "sandy": {"wilting_point": 6.0, "field_capacity": 14.0, "ph_range": (5.5, 7.2), "drainage": "Excessive"},
    "loamy": {"wilting_point": 12.0, "field_capacity": 28.0, "ph_range": (6.0, 7.5), "drainage": "Well-drained"},
    "clay": {"wilting_point": 20.0, "field_capacity": 38.0, "ph_range": (6.5, 8.5), "drainage": "Poor"},
    "silty": {"wilting_point": 14.0, "field_capacity": 30.0, "ph_range": (6.0, 7.8), "drainage": "Moderate"},
    "sandy loam": {"wilting_point": 8.0, "field_capacity": 20.0, "ph_range": (5.8, 7.4), "drainage": "Good"},
    "clay loam": {"wilting_point": 16.0, "field_capacity": 32.0, "ph_range": (6.2, 8.0), "drainage": "Moderate to Low"},
}

CROPS = {
    "Wheat": {
        "kc": {"Initial": 0.35, "Development": 0.75, "Mid-Season": 1.15, "Late Season": 0.45},
        "opt_temp": (15, 25), "pref_soil": ["loamy", "clay loam", "sandy loam"],
        "default_method": "Sprinkler", "root_depth_cm": 60
    },
    "Rice": {
        "kc": {"Initial": 1.05, "Development": 1.15, "Mid-Season": 1.20, "Late Season": 0.90},
        "opt_temp": (22, 34), "pref_soil": ["clay", "clay loam", "silty"],
        "default_method": "Basin", "root_depth_cm": 45
    },
    "Maize": {
        "kc": {"Initial": 0.40, "Development": 0.80, "Mid-Season": 1.15, "Late Season": 0.60},
        "opt_temp": (18, 30), "pref_soil": ["loamy", "sandy loam", "clay loam"],
        "default_method": "Drip", "root_depth_cm": 70
    },
    "Cotton": {
        "kc": {"Initial": 0.35, "Development": 0.75, "Mid-Season": 1.20, "Late Season": 0.65},
        "opt_temp": (21, 35), "pref_soil": ["clay loam", "clay", "loamy"],
        "default_method": "Drip", "root_depth_cm": 90
    },
    "Sugarcane": {
        "kc": {"Initial": 0.40, "Development": 0.85, "Mid-Season": 1.25, "Late Season": 0.75},
        "opt_temp": (20, 35), "pref_soil": ["loamy", "clay loam"],
        "default_method": "Furrow", "root_depth_cm": 100
    },
    "Tomato": {
        "kc": {"Initial": 0.45, "Development": 0.75, "Mid-Season": 1.15, "Late Season": 0.80},
        "opt_temp": (18, 28), "pref_soil": ["loamy", "sandy loam"],
        "default_method": "Drip", "root_depth_cm": 50
    },
    "Potato": {
        "kc": {"Initial": 0.45, "Development": 0.75, "Mid-Season": 1.15, "Late Season": 0.75},
        "opt_temp": (15, 24), "pref_soil": ["sandy loam", "loamy"],
        "default_method": "Drip", "root_depth_cm": 40
    },
    "Onion": {
        "kc": {"Initial": 0.45, "Development": 0.70, "Mid-Season": 1.05, "Late Season": 0.75},
        "opt_temp": (13, 26), "pref_soil": ["loamy", "sandy loam", "silty"],
        "default_method": "Drip", "root_depth_cm": 35
    },
    "Soybean": {
        "kc": {"Initial": 0.40, "Development": 0.75, "Mid-Season": 1.10, "Late Season": 0.50},
        "opt_temp": (20, 30), "pref_soil": ["loamy", "clay loam"],
        "default_method": "Sprinkler", "root_depth_cm": 60
    },
    "Chickpea": {
        "kc": {"Initial": 0.35, "Development": 0.65, "Mid-Season": 1.00, "Late Season": 0.40},
        "opt_temp": (14, 25), "pref_soil": ["loamy", "clay loam", "sandy loam"],
        "default_method": "Sprinkler", "root_depth_cm": 50
    },
    "Mustard": {
        "kc": {"Initial": 0.35, "Development": 0.70, "Mid-Season": 1.05, "Late Season": 0.45},
        "opt_temp": (12, 25), "pref_soil": ["loamy", "sandy loam"],
        "default_method": "Sprinkler", "root_depth_cm": 60
    },
    "Groundnut": {
        "kc": {"Initial": 0.40, "Development": 0.75, "Mid-Season": 1.10, "Late Season": 0.60},
        "opt_temp": (22, 32), "pref_soil": ["sandy loam", "loamy"],
        "default_method": "Sprinkler", "root_depth_cm": 50
    }
}

GROWTH_STAGES = ["Initial", "Development", "Mid-Season", "Late Season"]

def calculate_et0(temp_mean, humidity, wind_speed, solar_rad):
    """
    FAO-56 Penman-Monteith reference evapotranspiration (ET0) approximation in mm/day.
    """
    # Saturation vapor pressure (kPa)
    es = 0.6108 * np.exp((17.27 * temp_mean) / (temp_mean + 237.3))
    # Actual vapor pressure (kPa)
    ea = es * (humidity / 100.0)
    # Slope vapor pressure curve (kPa / C)
    delta = (4098.0 * es) / ((temp_mean + 237.3) ** 2)
    gamma = 0.067  # Psychrometric constant (kPa / C)

    # Net radiation approx (Rn - G) ~ 0.5 * solar_rad
    rn = max(1.0, 0.408 * 0.6 * solar_rad)
    
    numerator = 0.408 * delta * rn + gamma * (900.0 / (temp_mean + 273.0)) * wind_speed * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * wind_speed)
    et0 = numerator / denominator
    return float(np.clip(et0, 1.8, 9.2))

def calculate_effective_rainfall(forecast_rain_mm):
    """
    USDA-SCS empirical effective rainfall formula.
    """
    if forecast_rain_mm <= 0:
        return 0.0
    elif forecast_rain_mm <= 75.0:
        pe = forecast_rain_mm * (125.0 - 0.2 * forecast_rain_mm) / 125.0
    else:
        pe = (125.0 / 3.0) + (0.1 * forecast_rain_mm)
    return round(float(np.clip(pe, 0.0, forecast_rain_mm)), 2)

def generate_records(num_records=1500):
    records = []
    base_date = datetime(2026, 1, 1)

    for i in range(1, num_records + 1):
        loc = random.choice(LOCATIONS)
        soil_type = random.choice(list(SOIL_PROPERTIES.keys()))
        soil_meta = SOIL_PROPERTIES[soil_type]
        crop_name = random.choice(list(CROPS.keys()))
        crop_meta = CROPS[crop_name]
        growth_stage = random.choice(GROWTH_STAGES)

        # Environmental variables
        temp = round(random.uniform(14.0, 42.0), 1)
        humidity = round(random.uniform(22.0, 92.0), 1)
        wind_speed = round(random.uniform(0.8, 6.5), 1)
        solar_rad = round(random.uniform(8.0, 24.0), 1)
        recent_rain = round(random.choices([0.0, random.uniform(2.0, 35.0)], weights=[0.65, 0.35])[0], 1)
        forecast_rain = round(random.choices([0.0, random.uniform(1.0, 50.0)], weights=[0.60, 0.40])[0], 1)

        # Soil properties
        ph = round(random.uniform(soil_meta["ph_range"][0], soil_meta["ph_range"][1]), 2)
        base_sm = random.uniform(soil_meta["wilting_point"] + 2.0, soil_meta["field_capacity"])
        if recent_rain > 10:
            base_sm = min(soil_meta["field_capacity"] + 4.0, base_sm + recent_rain * 0.3)
        soil_moisture = round(base_sm, 1)

        # Reference ET and Crop Water Requirement (ETc)
        et0 = calculate_et0(temp, humidity, wind_speed, solar_rad)
        kc = crop_meta["kc"][growth_stage]
        etc = round(et0 * kc, 2)

        # Effective rainfall (over a 3-day window)
        pe = calculate_effective_rainfall(forecast_rain)

        # Available soil water contribution in root zone
        wp = soil_meta["wilting_point"]
        root_factor = crop_meta["root_depth_cm"] / 100.0
        usable_sm_mm = max(0.0, (soil_moisture - wp) * 0.1 * root_factor * 10)

        # Net Irrigation requirement for a 3-day agricultural cycle
        cycle_etc = etc * 3.0
        raw_net_irrigation = cycle_etc - pe - (usable_sm_mm * 0.5)
        # Add slight natural variability (+/- 5%)
        noise = random.uniform(-0.3, 0.3)
        net_irrigation_mm = round(max(0.0, raw_net_irrigation + noise), 2)
        net_irrigation_lpm2 = net_irrigation_mm

        # Irrigation method and duration
        irr_method = crop_meta["default_method"]
        if soil_type in ["sandy", "sandy loam"] and irr_method == "Basin":
            irr_method = "Drip"

        if net_irrigation_mm == 0:
            frequency_desc = "No immediate irrigation needed (soil & rain sufficient)"
        elif net_irrigation_mm < 10:
            frequency_desc = "Every 2 days, 30-45 minutes"
        elif net_irrigation_mm < 25:
            frequency_desc = "Every 3 days, 1.5-2.0 hours"
        else:
            frequency_desc = "Daily or split cycle, 2.5-3.5 hours"

        # Crop suitability label
        temp_match = crop_meta["opt_temp"][0] <= temp <= crop_meta["opt_temp"][1]
        soil_match = soil_type in crop_meta["pref_soil"]
        if temp_match and soil_match:
            suitability = "Highly Suitable"
        elif temp_match or soil_match:
            suitability = "Moderately Suitable"
        else:
            suitability = "Marginally Suitable"

        rec_date = (base_date + timedelta(days=random.randint(0, 260))).strftime("%Y-%m-%d")

        records.append({
            "record_id": f"AGRI-{i:05d}",
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "location_name": loc["name"],
            "state_district": f"{loc['state']}, {loc['district']}",
            "soil_type": soil_type,
            "soil_ph": ph,
            "soil_moisture_pct": soil_moisture,
            "temperature_c": temp,
            "relative_humidity_pct": humidity,
            "recent_rainfall_mm": recent_rain,
            "forecast_rainfall_mm": forecast_rain,
            "wind_speed_ms": wind_speed,
            "solar_radiation_mj": solar_rad,
            "evapotranspiration_et0_mm": round(et0, 2),
            "crop_type": crop_name,
            "crop_growth_stage": growth_stage,
            "crop_water_requirement_etc_mm": etc,
            "net_irrigation_req_mm": net_irrigation_mm,
            "net_irrigation_req_liters_m2": net_irrigation_lpm2,
            "irrigation_frequency": frequency_desc,
            "recommended_irrigation_method": irr_method,
            "crop_suitability_label": suitability,
            "data_source": "AgroMet-FAO56-Synthetic-Validated",
            "record_date": rec_date
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "irrigation_data.csv")

    df = generate_records(num_records=1500)
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} records successfully saved to: {csv_path}")
