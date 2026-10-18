from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.extensions import db
from app.models.advisory import AgriculturalAdvisory
from app.models.user import User

advisories_bp = Blueprint("advisories", __name__, url_prefix="/api/v1/advisories")

@advisories_bp.route("", methods=["GET"])
def list_advisories():
    crop = request.args.get("crop")
    soil = request.args.get("soil")
    category = request.args.get("category")

    query = AgriculturalAdvisory.query
    if crop:
        query = query.filter((AgriculturalAdvisory.target_crop == None) | (AgriculturalAdvisory.target_crop.ilike(f"%{crop}%")))
    if soil:
        query = query.filter((AgriculturalAdvisory.target_soil == None) | (AgriculturalAdvisory.target_soil == soil.lower()))
    if category:
        query = query.filter_by(category=category)

    results = query.order_by(AgriculturalAdvisory.is_emergency.desc(), AgriculturalAdvisory.created_at.desc()).all()
    return jsonify({"advisories": [a.to_dict() for a in results]}), 200

@advisories_bp.route("", methods=["POST"])
@jwt_required()
def create_advisory():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    if not user or user.role not in ["specialist", "admin"]:
        return jsonify({"error": "Only agricultural specialists or administrators can publish advisories."}), 403

    data = request.get_json() or {}
    title = data.get("title", "").strip()
    advisory_text = data.get("advisory_text", "").strip()

    if not title or not advisory_text:
        return jsonify({"error": "Title and advisory text are required."}), 400

    advisory = AgriculturalAdvisory(
        title=title,
        category=data.get("category", "General Advisory"),
        target_crop=data.get("target_crop"),
        target_soil=data.get("target_soil"),
        target_growth_stage=data.get("target_growth_stage"),
        advisory_text=advisory_text,
        precautions=data.get("precautions"),
        is_emergency=bool(data.get("is_emergency", False)),
        is_specialist_verified=True
    )
    db.session.add(advisory)
    db.session.commit()

    return jsonify({"message": "Advisory published successfully.", "advisory": advisory.to_dict()}), 201
