from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from app.extensions import db
from app.models.user import User, FarmerProfile, SpecialistProfile, RetailerProfile
from app.models.system import AuditLog

auth_bp = Blueprint("auth", __name__, url_prefix="/api/v1/auth")

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    full_name = data.get("full_name", "").strip()
    role = data.get("role", "farmer").strip().lower()
    phone = data.get("phone", "").strip()

    if not email or not password or not full_name:
        return jsonify({"error": "Email, password, and full name are required."}), 400

    if role not in ["farmer", "retailer", "customer", "specialist", "admin"]:
        return jsonify({"error": f"Invalid role '{role}'."}), 400

    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters long."}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "User with this email already exists."}), 409

    # Prevent direct registration of admin without existing admin secret
    if role == "admin":
        admin_secret = data.get("admin_secret")
        if admin_secret != "AGRI_SUPER_ADMIN_2026":
            return jsonify({"error": "Unauthorized to register as administrator."}), 403

    user = User(
        email=email,
        full_name=full_name,
        role=role,
        phone=phone
    )
    user.set_password(password)
    db.session.add(user)
    db.session.flush()

    # Create associated profile
    if role == "farmer":
        profile = FarmerProfile(
            user_id=user.id,
            farm_name=data.get("farm_name"),
            district=data.get("district"),
            state=data.get("state"),
            preferred_language=data.get("preferred_language", "English")
        )
        db.session.add(profile)
    elif role == "specialist":
        profile = SpecialistProfile(
            user_id=user.id,
            specialization=data.get("specialization", "General Agronomy"),
            qualification=data.get("qualification"),
            years_experience=int(data.get("years_experience", 1)),
            organization=data.get("organization")
        )
        db.session.add(profile)
    elif role == "retailer":
        profile = RetailerProfile(
            user_id=user.id,
            shop_name=data.get("shop_name", f"{full_name} Agri Supplies"),
            gst_number=data.get("gst_number"),
            address=data.get("address"),
            city=data.get("city"),
            state=data.get("state")
        )
        db.session.add(profile)

    # Log audit
    log = AuditLog(
        user_id=user.id,
        action="USER_REGISTRATION",
        resource="User",
        ip_address=request.remote_addr,
        details=f"New user registered with role: {role}"
    )
    db.session.add(log)
    db.session.commit()

    token = create_access_token(identity=str(user.id))
    return jsonify({
        "message": "Registration successful.",
        "access_token": token,
        "user": user.to_dict()
    }), 201

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid email or password."}), 401

    if not user.is_active:
        return jsonify({"error": "This account has been deactivated. Please contact support."}), 403

    token = create_access_token(identity=str(user.id))

    # Log audit
    log = AuditLog(
        user_id=user.id,
        action="USER_LOGIN",
        resource="Auth",
        ip_address=request.remote_addr,
        details=f"User {user.email} logged in successfully."
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({
        "message": "Login successful.",
        "access_token": token,
        "user": user.to_dict()
    }), 200

@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def get_me():
    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))
    if not user:
        return jsonify({"error": "User not found."}), 404
    return jsonify({"user": user.to_dict()}), 200

@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    # In stateless JWT, frontend clears token. Backend can record logout audit.
    user_id = get_jwt_identity()
    log = AuditLog(
        user_id=int(user_id),
        action="USER_LOGOUT",
        resource="Auth",
        ip_address=request.remote_addr,
        details="User logged out."
    )
    db.session.add(log)
    db.session.commit()
    return jsonify({"message": "Logout recorded successfully."}), 200
