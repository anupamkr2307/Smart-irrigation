from functools import wraps
from flask import jsonify
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
from app.models.user import User

def role_required(*allowed_roles):
    """
    Decorator to enforce Role-Based Access Control (RBAC).
    Usage:
        @jwt_required()
        @role_required("admin", "specialist")
        def route(): ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            verify_jwt_in_request()
            user_id = get_jwt_identity()
            # JWT identity might be string or int
            try:
                user_id = int(user_id)
            except (ValueError, TypeError):
                pass

            user = User.query.get(user_id)
            if not user or not user.is_active:
                return jsonify({"error": "Account not found or deactivated."}), 401

            if allowed_roles and user.role not in allowed_roles and user.role != "admin":
                return jsonify({
                    "error": "Forbidden: insufficient permissions for this resource.",
                    "required_roles": list(allowed_roles),
                    "current_role": user.role
                }), 403

            return fn(*args, **kwargs)
        return wrapper
    return decorator
