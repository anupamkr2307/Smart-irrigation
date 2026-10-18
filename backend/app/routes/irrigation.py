from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.extensions import db
from app.models.irrigation import IrrigationPrediction, CropReference
from app.models.system import AuditLog
from app.services.ml_service import ml_service
from app.services.water_service import calculate_water_balance, calculate_penman_et0, calculate_usda_effective_rain
from app.utils.validators import validate_irrigation_inputs

irrigation_bp = Blueprint("irrigation", __name__, url_prefix="/api/v1/irrigation")

@irrigation_bp.route("/predict", methods=["POST"])
@jwt_required()
def predict_irrigation():
    user_id = int(get_jwt_identity())
    data = request.get_json() or {}

    # 1. Validation
    errors = validate_irrigation_inputs(data)
    if errors:
        return jsonify({"errors": errors}), 400

    soil_type = data.get("soil_type", "loamy").lower().strip()
    crop_type = data.get("crop_type", "Wheat").strip()
    crop_growth_stage = data.get("crop_growth_stage", "Mid-Season").strip()
    farm_area = float(data.get("farm_area", 1.0))
    area_unit = data.get("area_unit", "acres").lower()
    location_name = data.get("location_name", "Farm Location").strip()

    # Weather & environmental parameters
    temp_c = float(data.get("temperature_c", 25.0))
    humidity_pct = float(data.get("relative_humidity_pct", 50.0))
    soil_moisture_pct = float(data.get("soil_moisture_pct", 25.0))
    recent_rain = float(data.get("recent_rainfall_mm", 0.0))
    forecast_rain = float(data.get("forecast_rainfall_mm", 0.0))
    wind_speed = float(data.get("wind_speed_ms", 2.2))
    solar_rad = float(data.get("solar_radiation_mj", 16.5))

    # Reference ET0 and crop Kc lookup
    et0 = float(data.get("evapotranspiration_et0_mm", 0.0))
    if et0 <= 0:
        et0 = calculate_penman_et0(temp_c, humidity_pct, wind_speed, solar_rad)

    crop_ref = CropReference.query.filter(CropReference.name.ilike(crop_type)).first()
    kc = 1.0
    recommended_method = "Drip"
    if crop_ref:
        stage_map = {
            "Initial": crop_ref.kc_initial,
            "Development": crop_ref.kc_development,
            "Mid-Season": crop_ref.kc_mid_season,
            "Late Season": crop_ref.kc_late_season
        }
        kc = stage_map.get(crop_growth_stage, 1.0)
        recommended_method = crop_ref.default_irrigation_method

    # 2. Run ML Pipeline for Continuous Net Irrigation (mm)
    ml_features = {
        "soil_type": soil_type,
        "crop_type": crop_type,
        "crop_growth_stage": crop_growth_stage,
        "temperature_c": temp_c,
        "relative_humidity_pct": humidity_pct,
        "soil_moisture_pct": soil_moisture_pct,
        "recent_rainfall_mm": recent_rain,
        "forecast_rainfall_mm": forecast_rain,
        "wind_speed_ms": wind_speed,
        "solar_radiation_mj": solar_rad,
        "evapotranspiration_et0_mm": et0
    }

    ml_result = ml_service.predict(ml_features)
    predicted_net_mm = ml_result["predicted_net_mm"]

    # 3. Transparent Water Balance & Gross Calculations
    water_balance = calculate_water_balance(
        crop_kc=kc,
        et0_mm=et0,
        forecast_rain_mm=forecast_rain,
        soil_type=soil_type,
        soil_moisture_pct=soil_moisture_pct,
        farm_area=farm_area,
        area_unit=area_unit,
        irrigation_method=recommended_method
    )

    # Use the ML prediction for net irrigation depth, and compute corresponding volume
    area_sqm = water_balance["area_sqm"]
    net_liters = round(predicted_net_mm * area_sqm, 1)
    efficiency = water_balance["irrigation_efficiency_pct"] / 100.0
    gross_mm = round(predicted_net_mm / efficiency, 2) if predicted_net_mm > 0 else 0.0
    gross_liters = round(gross_mm * area_sqm, 1)

    # 4. Save Prediction Record in Database
    prediction_record = IrrigationPrediction(
        user_id=user_id,
        farm_id=data.get("farm_id"),
        plot_id=data.get("plot_id"),
        location_name=location_name,
        soil_type=soil_type,
        crop_type=crop_type,
        crop_growth_stage=crop_growth_stage,
        farm_area=farm_area,
        area_unit=area_unit,
        temperature_c=temp_c,
        relative_humidity_pct=humidity_pct,
        soil_moisture_pct=soil_moisture_pct,
        recent_rainfall_mm=recent_rain,
        forecast_rainfall_mm=forecast_rain,
        wind_speed_ms=wind_speed,
        solar_radiation_mj=solar_rad,
        et0_mm=et0,
        etc_mm=water_balance["daily_etc_mm"],
        effective_rainfall_mm=water_balance["effective_rainfall_mm"],
        predicted_net_mm=predicted_net_mm,
        predicted_net_liters=net_liters,
        gross_irrigation_mm=gross_mm,
        gross_irrigation_liters=gross_liters,
        irrigation_efficiency_pct=water_balance["irrigation_efficiency_pct"],
        irrigation_frequency=water_balance["schedule"],
        recommended_irrigation_method=recommended_method,
        model_version=ml_result["model_version"],
        confidence_notes=f"MAE: ±{ml_result['mae_confidence_mm']} mm, R²: {ml_result['r2_score']:.2f}",
        water_saving_tips="\n".join(water_balance["water_saving_tips"]),
        soil_precautions=water_balance["soil_precautions"]
    )
    db.session.add(prediction_record)

    # Audit log
    audit = AuditLog(
        user_id=user_id,
        action="PREDICT_IRRIGATION",
        resource="IrrigationPrediction",
        ip_address=request.remote_addr,
        details=f"Prediction generated for {crop_type} ({farm_area} {area_unit}): {predicted_net_mm} mm ({net_liters} L)"
    )
    db.session.add(audit)
    db.session.commit()

    return jsonify({
        "id": prediction_record.id,
        "location_name": location_name,
        "soil_type": soil_type,
        "crop_type": crop_type,
        "crop_growth_stage": crop_growth_stage,
        "farm_area": farm_area,
        "area_unit": area_unit,
        "area_sq_meters": area_sqm,
        "predictions": {
            "net_irrigation_depth_mm": predicted_net_mm,
            "net_irrigation_volume_liters": net_liters,
            "gross_irrigation_depth_mm": gross_mm,
            "gross_irrigation_volume_liters": gross_liters,
            "effective_rainfall_depth_mm": water_balance["effective_rainfall_mm"],
            "effective_rainfall_volume_liters": water_balance["effective_rainfall_liters"],
            "forecast_rainfall_depth_mm": forecast_rain,
            "forecast_rainfall_volume_liters": water_balance["forecast_rainfall_liters"],
            "crop_evapotranspiration_etc_mm_day": water_balance["daily_etc_mm"],
            "reference_et0_mm_day": round(et0, 2),
            "irrigation_efficiency_pct": water_balance["irrigation_efficiency_pct"]
        },
        "advisory": {
            "suggested_schedule": water_balance["schedule"],
            "recommended_method": recommended_method,
            "water_saving_suggestions": water_balance["water_saving_tips"],
            "soil_precautions": water_balance["soil_precautions"]
        },
        "ml_diagnostics": {
            "model_version": ml_result["model_version"],
            "r2_score": ml_result["r2_score"],
            "mae_uncertainty_mm": ml_result["mae_confidence_mm"],
            "confidence_interval_mm": ml_result["uncertainty_interval_mm"],
            "main_factors": ml_result["main_factors"]
        }
    }), 200

@irrigation_bp.route("/history", methods=["GET"])
@jwt_required()
def prediction_history():
    user_id = int(get_jwt_identity())
    records = IrrigationPrediction.query.filter_by(user_id=user_id).order_by(IrrigationPrediction.created_at.desc()).limit(50).all()
    return jsonify({"history": [r.to_dict() for r in records]}), 200
