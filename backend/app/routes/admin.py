import os
import json
import pandas as pd
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.extensions import db
from app.models.user import User, SpecialistProfile
from app.models.marketplace import Order, Product
from app.models.irrigation import IrrigationPrediction, MLModelVersion
from app.models.advisory import Consultation
from app.models.system import AuditLog
from app.utils.decorators import role_required

admin_bp = Blueprint("admin", __name__, url_prefix="/api/v1/admin")

@admin_bp.route("/analytics", methods=["GET"])
@jwt_required()
@role_required("admin")
def get_analytics():
    total_users = User.query.count()
    farmers_count = User.query.filter_by(role="farmer").count()
    retailers_count = User.query.filter_by(role="retailer").count()
    specialists_count = User.query.filter_by(role="specialist").count()
    customers_count = User.query.filter_by(role="customer").count()

    total_predictions = IrrigationPrediction.query.count()
    total_consultations = Consultation.query.count()
    total_orders = Order.query.count()
    total_revenue = db.session.query(db.func.sum(Order.total_amount)).scalar() or 0.0

    # Estimate water saved: difference between flood irrigation (50% efficiency) and precision drip (90% efficiency)
    total_gross_liters = db.session.query(db.func.sum(IrrigationPrediction.gross_irrigation_liters)).scalar() or 0.0
    estimated_water_saved_liters = round(total_gross_liters * 0.35, 1)

    return jsonify({
        "users": {
            "total": total_users,
            "farmers": farmers_count,
            "retailers": retailers_count,
            "specialists": specialists_count,
            "customers": customers_count
        },
        "operations": {
            "irrigation_predictions_run": total_predictions,
            "consultations_logged": total_consultations,
            "marketplace_orders": total_orders,
            "total_marketplace_revenue_inr": round(total_revenue, 2),
            "estimated_water_saved_liters": estimated_water_saved_liters
        }
    }), 200

@admin_bp.route("/users", methods=["GET"])
@jwt_required()
@role_required("admin")
def list_users():
    role_filter = request.args.get("role")
    query = User.query
    if role_filter:
        query = query.filter_by(role=role_filter)

    users = query.order_by(User.created_at.desc()).all()
    return jsonify({"users": [u.to_dict() for u in users]}), 200

@admin_bp.route("/users/<int:user_id>/status", methods=["PATCH"])
@jwt_required()
@role_required("admin")
def toggle_user_status(user_id):
    admin_id = int(get_jwt_identity())
    user = User.query.get_or_404(user_id)

    if user.id == admin_id:
        return jsonify({"error": "Admin cannot deactivate their own account."}), 400

    data = request.get_json() or {}
    is_active = data.get("is_active", not user.is_active)
    user.is_active = bool(is_active)

    log = AuditLog(
        user_id=admin_id,
        action="TOGGLE_USER_STATUS",
        resource="User",
        ip_address=request.remote_addr,
        details=f"User {user.email} status set to {'active' if is_active else 'deactivated'}."
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({"message": f"User status updated.", "user": user.to_dict()}), 200

@admin_bp.route("/specialists/verify/<int:user_id>", methods=["POST"])
@jwt_required()
@role_required("admin")
def verify_specialist(user_id):
    admin_id = int(get_jwt_identity())
    user = User.query.get_or_404(user_id)
    if user.role != "specialist" or not user.specialist_profile:
        return jsonify({"error": "Target user is not a specialist."}), 400

    data = request.get_json() or {}
    verified = bool(data.get("is_verified", True))
    user.specialist_profile.is_verified = verified

    log = AuditLog(
        user_id=admin_id,
        action="VERIFY_SPECIALIST",
        resource="SpecialistProfile",
        ip_address=request.remote_addr,
        details=f"Specialist {user.full_name} verification status set to {verified}."
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({"message": "Specialist verification updated.", "user": user.to_dict()}), 200

@admin_bp.route("/ml-metrics", methods=["GET"])
@jwt_required()
@role_required("admin")
def get_ml_metrics():
    metrics_path = current_app.config.get("ML_METRICS_PATH")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return jsonify({"ml_metrics": metrics}), 200
    return jsonify({"error": "ML metrics file not found."}), 404

@admin_bp.route("/dataset-sample", methods=["GET"])
@jwt_required()
@role_required("admin")
def get_dataset_sample():
    dataset_path = current_app.config.get("DATASET_PATH")
    if not os.path.exists(dataset_path):
        return jsonify({"error": "Dataset CSV file not found."}), 404

    df = pd.read_csv(dataset_path)
    sample_records = df.head(30).to_dict(orient="records")
    return jsonify({
        "total_records": len(df),
        "columns": list(df.columns),
        "sample": sample_records
    }), 200

@admin_bp.route("/audit-logs", methods=["GET"])
@jwt_required()
@role_required("admin")
def get_audit_logs():
    logs = AuditLog.query.order_by(AuditLog.created_at.desc()).limit(100).all()
    return jsonify({"audit_logs": [l.to_dict() for l in logs]}), 200
