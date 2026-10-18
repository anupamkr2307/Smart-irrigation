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
        farm_name = data.get("farm_name") or f"{full_name}'s Farm"
        district = data.get("district") or "Nashik"
        state = data.get("state") or "Maharashtra"
        profile = FarmerProfile(
            user_id=user.id,
            farm_name=farm_name,
            district=district,
            state=state,
            preferred_language=data.get("preferred_language", "English"),
            primary_crops=data.get("primary_crops", "Tomato, Wheat")
        )
        db.session.add(profile)

        # Create default Farm entry for immediate usability
        try:
            from app.models.farm import Farm, FarmPlot
            total_area = float(data.get("total_area", 5.0))
            soil_type = (data.get("soil_type") or "loamy").lower()
            farm = Farm(
                farmer_id=user.id,
                name=farm_name,
                total_area=total_area,
                area_unit=data.get("area_unit", "acres"),
                location_name=f"{district}, {state}",
                district=district,
                state=state,
                latitude=float(data.get("latitude", 19.9975)),
                longitude=float(data.get("longitude", 73.7898))
            )
            db.session.add(farm)
            db.session.flush()
            plot = FarmPlot(
                farm_id=farm.id,
                plot_name="Plot 1 (Main Field)",
                area=total_area,
                soil_type=soil_type,
                current_crop=data.get("primary_crop", "Tomato"),
                irrigation_system=data.get("irrigation_system", "Drip")
            )
            db.session.add(plot)
        except Exception as e:
            print(f"Farm creation note: {e}")

    elif role == "specialist":
        profile = SpecialistProfile(
            user_id=user.id,
            specialization=data.get("specialization", "General Agronomy"),
            qualification=data.get("qualification", "B.Sc. Agriculture"),
            years_experience=int(data.get("years_experience", 1)),
            license_number=data.get("license_number"),
            organization=data.get("organization"),
            bio=data.get("bio")
        )
        db.session.add(profile)
    elif role == "retailer":
        profile = RetailerProfile(
            user_id=user.id,
            shop_name=data.get("shop_name", f"{full_name} Agri Supplies"),
            gst_number=data.get("gst_number"),
            license_number=data.get("license_number"),
            address=data.get("address"),
            city=data.get("city", "Nashik"),
            state=data.get("state", "Maharashtra"),
            delivery_supported=bool(data.get("delivery_supported", True))
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
        "user": user.to_dict(),
        "permissions": user.get_permissions()
    }), 201

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    target_role = data.get("target_role", "").strip().lower()

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid email or password."}), 401

    if not user.is_active:
        return jsonify({"error": "This account has been deactivated. Please contact support."}), 403

    # Enforce role-based permission verification when logging into a specific portal
    if target_role and target_role != "all":
        if user.role != target_role and user.role != "admin":
            return jsonify({
                "error": f"Permission Denied: Your account is registered as a '{user.role.capitalize()}' and does not have access to the {target_role.capitalize()} Portal.",
                "user_role": user.role,
                "target_role": target_role
            }), 403

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
        "user": user.to_dict(),
        "permissions": user.get_permissions()
    }), 200

@auth_bp.route("/roles", methods=["GET"])
def get_roles():
    from app.models.user import ROLE_PERMISSIONS
    role_info = {
        "farmer": {
            "name": "Farmer",
            "title": "Agricultural Producer & Cultivator",
            "description": "Register your farm holdings, predict precise crop water needs via FAO-56 Penman-Monteith AI, get seasonal crop suitability advice, and connect with certified agronomy specialists.",
            "permissions": ROLE_PERMISSIONS["farmer"],
            "dashboard_url": "/farmer/dashboard"
        },
        "retailer": {
            "name": "Retailer",
            "title": "Agri-Input & Equipment Merchant",
            "description": "List drip irrigation equipment, fertilizers, seeds, and machinery. Manage inventory, fulfill customer orders, and submit competitive bids on farmer quotation requests.",
            "permissions": ROLE_PERMISSIONS["retailer"],
            "dashboard_url": "/retailer/dashboard"
        },
        "specialist": {
            "name": "Specialist",
            "title": "Certified Agronomist & Soil Scientist",
            "description": "Provide certified agricultural advisory, diagnose crop diseases and pest outbreaks through farmer consultation tickets, and publish verified agronomic guidance.",
            "permissions": ROLE_PERMISSIONS["specialist"],
            "dashboard_url": "/specialist/dashboard"
        },
        "customer": {
            "name": "Customer",
            "title": "Produce Buyer & Agri-Consumer",
            "description": "Purchase farm equipment, organic inputs, and farm-fresh produce directly from verified sellers with secure checkout and order tracking.",
            "permissions": ROLE_PERMISSIONS["customer"],
            "dashboard_url": "/customer/dashboard"
        },
        "admin": {
            "name": "Administrator",
            "title": "Platform Governance & System Operations",
            "description": "Oversee platform RBAC permissions, verify agronomist credentials, monitor ML pipeline metrics, and inspect system audit logs.",
            "permissions": ROLE_PERMISSIONS["admin"],
            "dashboard_url": "/admin/dashboard"
        }
    }
    return jsonify({"roles": role_info}), 200


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
