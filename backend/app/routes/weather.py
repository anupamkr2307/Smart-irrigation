from flask import Blueprint, request, jsonify
from app.services.weather_service import geocode_location, fetch_weather_data
from app.models.farm import WeatherRecord
from app.extensions import db

weather_bp = Blueprint("weather", __name__, url_prefix="/api/v1/weather")

@weather_bp.route("/geocode", methods=["GET"])
def geocode():
    q = request.args.get("query", "").strip()
    if not q:
        return jsonify({"error": "Query parameter 'query' is required."}), 400
    res = geocode_location(q)
    return jsonify({"location": res}), 200

@weather_bp.route("/current", methods=["GET"])
def get_current_weather():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    loc_name = request.args.get("location_name", "Selected Region")

    if not lat or not lon:
        return jsonify({"error": "Query parameters 'lat' and 'lon' are required."}), 400

    try:
        latitude = float(lat)
        longitude = float(lon)
    except ValueError:
        return jsonify({"error": "Latitude and longitude must be valid floating numbers."}), 400

    weather = fetch_weather_data(latitude, longitude, loc_name)

    # Optional background recording
    try:
        record = WeatherRecord(
            location_name=loc_name,
            latitude=latitude,
            longitude=longitude,
            temperature_c=weather["temperature_c"],
            relative_humidity_pct=weather["relative_humidity_pct"],
            rainfall_recent_mm=weather["recent_rainfall_mm"],
            rainfall_forecast_mm=weather["forecast_rainfall_3d_mm"],
            wind_speed_ms=weather["wind_speed_ms"],
            solar_radiation_mj=weather["solar_radiation_mj"],
            et0_mm=weather["et0_mm_day"],
            source=weather.get("source", "Open-Meteo")
        )
        db.session.add(record)
        db.session.commit()
    except Exception:
        db.session.rollback()

    return jsonify({"weather": weather}), 200
