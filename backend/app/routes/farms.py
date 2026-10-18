from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.extensions import db
from app.models.farm import Farm, FarmPlot
from app.models.user import User

farms_bp = Blueprint("farms", __name__, url_prefix="/api/v1/farms")

@farms_bp.route("", methods=["GET"])
@jwt_required()
def list_farms():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404

    # If admin, can view all, otherwise only owned farms
    if user.role == "admin":
        farms = Farm.query.all()
    else:
        farms = Farm.query.filter_by(farmer_id=user_id).all()

    return jsonify({"farms": [f.to_dict() for f in farms]}), 200

@farms_bp.route("", methods=["POST"])
@jwt_required()
def create_farm():
    user_id = int(get_jwt_identity())
    data = request.get_json() or {}

    name = data.get("name", "").strip()
    location_name = data.get("location_name", "").strip()
    total_area = float(data.get("total_area", 1.0))
    area_unit = data.get("area_unit", "acres").lower()

    if not name or not location_name:
        return jsonify({"error": "Farm name and location are required."}), 400

    farm = Farm(
        farmer_id=user_id,
        name=name,
        total_area=total_area,
        area_unit=area_unit,
        location_name=location_name,
        state=data.get("state"),
        district=data.get("district"),
        latitude=float(data.get("latitude", 19.9975)),
        longitude=float(data.get("longitude", 73.7898))
    )
    db.session.add(farm)
    db.session.flush()

    # Optional initial plot
    initial_plots = data.get("plots", [])
    if initial_plots:
        for p in initial_plots:
            plot = FarmPlot(
                farm_id=farm.id,
                plot_name=p.get("plot_name", "Main Field"),
                area=float(p.get("area", total_area)),
                soil_type=p.get("soil_type", "loamy"),
                current_crop=p.get("current_crop", "Wheat"),
                growth_stage=p.get("growth_stage", "Mid-Season"),
                irrigation_system=p.get("irrigation_system", "Drip"),
                soil_ph=float(p.get("soil_ph", 6.8)),
                target_moisture_pct=float(p.get("target_moisture_pct", 25.0))
            )
            db.session.add(plot)
    else:
        # Create at least 1 default plot
        default_plot = FarmPlot(
            farm_id=farm.id,
            plot_name=f"{name} - Plot 1",
            area=total_area,
            soil_type=data.get("soil_type", "loamy"),
            current_crop=data.get("current_crop", "Tomato"),
            growth_stage="Mid-Season",
            irrigation_system="Drip"
        )
        db.session.add(default_plot)

    db.session.commit()
    return jsonify({"message": "Farm created successfully.", "farm": farm.to_dict()}), 201

@farms_bp.route("/<int:farm_id>", methods=["GET"])
@jwt_required()
def get_farm(farm_id):
    user_id = int(get_jwt_identity())
    farm = Farm.query.get_or_404(farm_id)
    if farm.farmer_id != user_id and User.query.get(user_id).role != "admin":
        return jsonify({"error": "Unauthorized access to this farm."}), 403
    return jsonify({"farm": farm.to_dict()}), 200

@farms_bp.route("/<int:farm_id>/plots", methods=["POST"])
@jwt_required()
def add_plot(farm_id):
    user_id = int(get_jwt_identity())
    farm = Farm.query.get_or_404(farm_id)
    if farm.farmer_id != user_id and User.query.get(user_id).role != "admin":
        return jsonify({"error": "Unauthorized access to this farm."}), 403

    data = request.get_json() or {}
    plot_name = data.get("plot_name", "").strip()
    if not plot_name:
        return jsonify({"error": "Plot name is required."}), 400

    plot = FarmPlot(
        farm_id=farm.id,
        plot_name=plot_name,
        area=float(data.get("area", 1.0)),
        soil_type=data.get("soil_type", "loamy"),
        current_crop=data.get("current_crop", "Wheat"),
        growth_stage=data.get("growth_stage", "Mid-Season"),
        irrigation_system=data.get("irrigation_system", "Drip"),
        soil_ph=float(data.get("soil_ph", 6.8)),
        target_moisture_pct=float(data.get("target_moisture_pct", 25.0))
    )
    db.session.add(plot)
    db.session.commit()

    return jsonify({"message": "Plot added successfully.", "plot": plot.to_dict()}), 201

@farms_bp.route("/<int:farm_id>", methods=["DELETE"])
@jwt_required()
def delete_farm(farm_id):
    user_id = int(get_jwt_identity())
    farm = Farm.query.get_or_404(farm_id)
    if farm.farmer_id != user_id and User.query.get(user_id).role != "admin":
        return jsonify({"error": "Unauthorized."}), 403

    db.session.delete(farm)
    db.session.commit()
    return jsonify({"message": "Farm deleted."}), 200
