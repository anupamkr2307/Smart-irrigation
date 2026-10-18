from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.extensions import db
from app.models.advisory import Consultation, ConsultationMessage
from app.models.user import User
from app.models.system import Notification

consultations_bp = Blueprint("consultations", __name__, url_prefix="/api/v1/consultations")

@consultations_bp.route("", methods=["GET"])
@jwt_required()
def list_consultations():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404

    status = request.args.get("status")
    query = Consultation.query

    if user.role == "farmer":
        query = query.filter_by(farmer_id=user_id)
    elif user.role == "specialist":
        # Specialists see either open consultations or consultations assigned to them
        if status == "open":
            query = query.filter_by(status="open")
        else:
            query = query.filter((Consultation.specialist_id == user_id) | (Consultation.status == "open"))
    # Admin sees all

    if status and user.role != "specialist":
        query = query.filter_by(status=status)

    consultations = query.order_by(Consultation.created_at.desc()).all()
    return jsonify({"consultations": [c.to_dict() for c in consultations]}), 200

@consultations_bp.route("", methods=["POST"])
@jwt_required()
def create_consultation():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404

    data = request.get_json() or {}
    subject = data.get("subject", "").strip()
    issue_category = data.get("issue_category", "Pest & Disease").strip()
    description = data.get("description", "").strip()

    if not subject or not description:
        return jsonify({"error": "Subject and description are required."}), 400

    consultation = Consultation(
        farmer_id=user_id,
        subject=subject,
        issue_category=issue_category,
        crop_name=data.get("crop_name"),
        soil_type=data.get("soil_type"),
        location=data.get("location"),
        description=description,
        image_url=data.get("image_url"),
        status="open"
    )
    db.session.add(consultation)
    db.session.flush()

    # Add initial message from farmer
    initial_msg = ConsultationMessage(
        consultation_id=consultation.id,
        sender_id=user_id,
        message=description,
        attachment_url=data.get("image_url"),
        is_specialist_advice=False
    )
    db.session.add(initial_msg)
    db.session.commit()

    return jsonify({
        "message": "Consultation request submitted successfully.",
        "consultation": consultation.to_dict()
    }), 201

@consultations_bp.route("/<int:consultation_id>", methods=["GET"])
@jwt_required()
def get_consultation(consultation_id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    consultation = Consultation.query.get_or_404(consultation_id)

    # Permission check: farmer owner, assigned specialist, any specialist if open, or admin
    if user.role == "farmer" and consultation.farmer_id != user_id:
        return jsonify({"error": "Unauthorized."}), 403

    return jsonify({"consultation": consultation.to_dict()}), 200

@consultations_bp.route("/<int:consultation_id>/messages", methods=["POST"])
@jwt_required()
def add_message(consultation_id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    consultation = Consultation.query.get_or_404(consultation_id)

    data = request.get_json() or {}
    msg_text = data.get("message", "").strip()
    if not msg_text:
        return jsonify({"error": "Message content cannot be empty."}), 400

    is_specialist = user.role in ["specialist", "admin"]
    
    # If a specialist replies to an open consultation, assign it to them
    if is_specialist and not consultation.specialist_id:
        consultation.specialist_id = user_id
        consultation.status = "in_progress"

    msg = ConsultationMessage(
        consultation_id=consultation.id,
        sender_id=user_id,
        message=msg_text,
        attachment_url=data.get("attachment_url"),
        is_specialist_advice=is_specialist
    )
    db.session.add(msg)

    # Notify counterpart
    target_user_id = consultation.farmer_id if is_specialist else consultation.specialist_id
    if target_user_id:
        notif = Notification(
            user_id=target_user_id,
            title="New Consultation Message",
            message=f"{user.full_name}: {msg_text[:60]}...",
            link=f"/farmer/consultations?id={consultation.id}"
        )
        db.session.add(notif)

    db.session.commit()
    return jsonify({"message": "Message sent successfully.", "new_message": msg.to_dict()}), 201

@consultations_bp.route("/<int:consultation_id>/status", methods=["PATCH"])
@jwt_required()
def update_status(consultation_id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    consultation = Consultation.query.get_or_404(consultation_id)

    if user.role not in ["specialist", "admin", "farmer"]:
        return jsonify({"error": "Unauthorized."}), 403

    data = request.get_json() or {}
    new_status = data.get("status")
    if new_status not in ["open", "assigned", "in_progress", "resolved", "closed"]:
        return jsonify({"error": "Invalid status."}), 400

    consultation.status = new_status
    db.session.commit()
    return jsonify({"message": f"Status updated to {new_status}.", "consultation": consultation.to_dict()}), 200

@consultations_bp.route("/<int:consultation_id>/rate", methods=["POST"])
@jwt_required()
def rate_consultation(consultation_id):
    user_id = int(get_jwt_identity())
    consultation = Consultation.query.get_or_404(consultation_id)

    if consultation.farmer_id != user_id:
        return jsonify({"error": "Only the farmer who requested consultation can rate it."}), 403

    data = request.get_json() or {}
    rating = int(data.get("rating", 5))
    if rating < 1 or rating > 5:
        return jsonify({"error": "Rating must be between 1 and 5."}), 400

    consultation.rating = rating
    consultation.feedback = data.get("feedback")
    consultation.status = "resolved"
    db.session.commit()

    return jsonify({"message": "Feedback submitted successfully.", "consultation": consultation.to_dict()}), 200
