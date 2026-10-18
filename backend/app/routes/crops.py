from flask import Blueprint, request, jsonify
from app.models.irrigation import CropReference
from app.services.crop_recommender import recommend_top_crops

crops_bp = Blueprint("crops", __name__, url_prefix="/api/v1/crops")

@crops_bp.route("", methods=["GET"])
def list_crops():
    crops = CropReference.query.all()
    return jsonify({"crops": [c.to_dict() for c in crops]}), 200

@crops_bp.route("/recommend", methods=["POST"])
def recommend_crops():
    data = request.get_json() or {}
    soil_type = data.get("soil_type", "loamy").lower().strip()
    soil_ph = float(data.get("soil_ph", 6.8))
    temperature_c = float(data.get("temperature_c", 26.0))
    humidity_pct = float(data.get("relative_humidity_pct", 50.0))
    water_availability = data.get("water_availability", "Medium") # Low, Medium, High

    recommendations = recommend_top_crops(
        soil_type=soil_type,
        soil_ph=soil_ph,
        current_temp=temperature_c,
        current_humidity=humidity_pct,
        water_availability=water_availability
    )

    return jsonify({
        "inputs": {
            "soil_type": soil_type,
            "soil_ph": soil_ph,
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "water_availability": water_availability
        },
        "recommendations": recommendations,
        "scoring_methodology": "Multi-criteria agronomic evaluation combining soil texture compatibility (35%), pH affinity (20%), temperature tolerance range (25%), and water supply feasibility (20%)."
    }), 200
