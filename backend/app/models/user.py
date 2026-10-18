from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from app.extensions import db

ROLE_PERMISSIONS = {
    "farmer": [
        "access_farmer_portal",
        "manage_farms",
        "calculate_irrigation",
        "view_crop_recommendations",
        "submit_consultations",
        "request_quotations",
        "browse_marketplace"
    ],
    "retailer": [
        "access_retailer_portal",
        "manage_inventory",
        "manage_orders",
        "respond_quotations",
        "view_retailer_metrics"
    ],
    "specialist": [
        "access_specialist_portal",
        "review_consultations",
        "provide_certified_advice",
        "publish_advisories",
        "manage_specialist_profile"
    ],
    "customer": [
        "access_customer_portal",
        "browse_marketplace",
        "create_orders",
        "view_order_history"
    ],
    "admin": [
        "access_admin_portal",
        "all",
        "manage_users",
        "verify_specialists",
        "view_ml_analytics",
        "view_audit_logs",
        "system_config"
    ]
}

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    role = db.Column(db.String(20), nullable=False, default="farmer") # farmer, retailer, customer, specialist, admin
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    farmer_profile = db.relationship("FarmerProfile", backref="user", uselist=False, cascade="all, delete-orphan")
    specialist_profile = db.relationship("SpecialistProfile", backref="user", uselist=False, cascade="all, delete-orphan")
    retailer_profile = db.relationship("RetailerProfile", backref="user", uselist=False, cascade="all, delete-orphan")
    farms = db.relationship("Farm", backref="owner", lazy=True, cascade="all, delete-orphan")
    predictions = db.relationship("IrrigationPrediction", backref="user", lazy=True)
    consultations_as_farmer = db.relationship("Consultation", foreign_keys="Consultation.farmer_id", backref="farmer", lazy=True)
    consultations_as_specialist = db.relationship("Consultation", foreign_keys="Consultation.specialist_id", backref="specialist", lazy=True)
    orders = db.relationship("Order", foreign_keys="Order.customer_id", backref="customer", lazy=True)
    retailer_orders = db.relationship("Order", foreign_keys="Order.retailer_id", lazy=True)
    notifications = db.relationship("Notification", backref="user", lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_permissions(self):
        return ROLE_PERMISSIONS.get(self.role, [])

    def to_dict(self):
        data = {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "phone": self.phone,
            "role": self.role,
            "permissions": self.get_permissions(),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
        if self.role == "farmer" and self.farmer_profile:
            data["profile"] = self.farmer_profile.to_dict()
        elif self.role == "specialist" and self.specialist_profile:
            data["profile"] = self.specialist_profile.to_dict()
        elif self.role == "retailer" and self.retailer_profile:
            data["profile"] = self.retailer_profile.to_dict()
        return data


class FarmerProfile(db.Model):
    __tablename__ = "farmer_profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    farm_name = db.Column(db.String(120), nullable=True)
    preferred_language = db.Column(db.String(20), default="English")
    district = db.Column(db.String(80), nullable=True)
    state = db.Column(db.String(80), nullable=True)
    primary_crops = db.Column(db.String(200), nullable=True)

    def to_dict(self):
        return {
            "farm_name": self.farm_name,
            "preferred_language": self.preferred_language,
            "district": self.district,
            "state": self.state,
            "primary_crops": self.primary_crops
        }

class SpecialistProfile(db.Model):
    __tablename__ = "specialist_profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    specialization = db.Column(db.String(120), nullable=False) # e.g., Agronomy, Soil Science, Pest Management
    qualification = db.Column(db.String(120), nullable=True)
    years_experience = db.Column(db.Integer, default=1)
    license_number = db.Column(db.String(80), nullable=True)
    organization = db.Column(db.String(150), nullable=True)
    is_verified = db.Column(db.Boolean, default=False)
    bio = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            "specialization": self.specialization,
            "qualification": self.qualification,
            "years_experience": self.years_experience,
            "license_number": self.license_number,
            "organization": self.organization,
            "is_verified": self.is_verified,
            "bio": self.bio
        }

class RetailerProfile(db.Model):
    __tablename__ = "retailer_profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    shop_name = db.Column(db.String(150), nullable=False)
    gst_number = db.Column(db.String(30), nullable=True)
    license_number = db.Column(db.String(60), nullable=True)
    address = db.Column(db.String(255), nullable=True)
    city = db.Column(db.String(80), nullable=True)
    state = db.Column(db.String(80), nullable=True)
    delivery_supported = db.Column(db.Boolean, default=True)

    def to_dict(self):
        return {
            "shop_name": self.shop_name,
            "gst_number": self.gst_number,
            "license_number": self.license_number,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "delivery_supported": self.delivery_supported
        }
