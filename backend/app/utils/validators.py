"""
Input validation and bounds checking for agricultural parameters
"""

VALID_SOIL_TYPES = ["sandy", "loamy", "clay", "silty", "sandy loam", "clay loam"]
VALID_GROWTH_STAGES = ["Initial", "Development", "Mid-Season", "Late Season"]

def validate_irrigation_inputs(data):
    errors = []

    soil_type = data.get("soil_type", "").lower().strip()
    if soil_type not in VALID_SOIL_TYPES:
        errors.append(f"Invalid soil type '{soil_type}'. Must be one of: {', '.join(VALID_SOIL_TYPES)}")

    crop_type = data.get("crop_type", "").strip()
    if not crop_type:
        errors.append("Crop type is required.")

    growth_stage = data.get("crop_growth_stage", "Mid-Season").strip()
    if growth_stage not in VALID_GROWTH_STAGES:
        errors.append(f"Invalid growth stage '{growth_stage}'. Must be one of: {', '.join(VALID_GROWTH_STAGES)}")

    try:
        farm_area = float(data.get("farm_area", 1.0))
        if farm_area <= 0 or farm_area > 50000:
            errors.append("Farm area must be greater than 0 and less than 50,000.")
    except (ValueError, TypeError):
        errors.append("Farm area must be a valid number.")

    try:
        temp = float(data.get("temperature_c", 25.0))
        if temp < -10 or temp > 60:
            errors.append("Temperature must be between -10°C and 60°C.")
    except (ValueError, TypeError):
        errors.append("Temperature must be a valid number.")

    try:
        humidity = float(data.get("relative_humidity_pct", 50.0))
        if humidity < 0 or humidity > 100:
            errors.append("Relative humidity must be between 0% and 100%.")
    except (ValueError, TypeError):
        errors.append("Relative humidity must be a valid number.")

    try:
        soil_moisture = float(data.get("soil_moisture_pct", 25.0))
        if soil_moisture < 0 or soil_moisture > 100:
            errors.append("Soil moisture must be between 0% and 100%.")
    except (ValueError, TypeError):
        errors.append("Soil moisture must be a valid number.")

    return errors
