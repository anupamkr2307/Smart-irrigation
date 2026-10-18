from datetime import datetime
from app.extensions import db

class Address(db.Model):
    __tablename__ = "addresses"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    label = db.Column(db.String(50), default="Home/Farm")
    line1 = db.Column(db.String(150), nullable=False)
    line2 = db.Column(db.String(150), nullable=True)
    city = db.Column(db.String(80), nullable=False)
    district = db.Column(db.String(80), nullable=True)
    state = db.Column(db.String(80), nullable=False)
    pincode = db.Column(db.String(20), nullable=False)
    is_default = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "line1": self.line1,
            "line2": self.line2,
            "city": self.city,
            "district": self.district,
            "state": self.state,
            "pincode": self.pincode,
            "is_default": self.is_default
        }

class Notification(db.Model):
    __tablename__ = "notifications"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    title = db.Column(db.String(150), nullable=False)
    message = db.Column(db.Text, nullable=False)
    link = db.Column(db.String(255), nullable=True)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "message": self.message,
            "link": self.link,
            "is_read": self.is_read,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class SpecialistVerification(db.Model):
    __tablename__ = "specialist_verifications"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    reviewed_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    status = db.Column(db.String(30), default="pending") # pending, approved, rejected
    document_url = db.Column(db.String(255), nullable=True)
    admin_notes = db.Column(db.Text, nullable=True)

    user = db.relationship("User", foreign_keys=[user_id])
    reviewer = db.relationship("User", foreign_keys=[reviewed_by])

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "specialist_name": self.user.full_name if self.user else "Unknown",
            "specialization": self.user.specialist_profile.specialization if (self.user and self.user.specialist_profile) else "N/A",
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewed_by_name": self.reviewer.full_name if self.reviewer else None,
            "status": self.status,
            "admin_notes": self.admin_notes
        }

class AuditLog(db.Model):
    __tablename__ = "audit_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    action = db.Column(db.String(80), nullable=False) # e.g., USER_LOGIN, ROLE_CHANGE, VERIFY_SPECIALIST, PREDICT_IRRIGATION
    resource = db.Column(db.String(120), nullable=True)
    ip_address = db.Column(db.String(50), nullable=True)
    details = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    user = db.relationship("User", foreign_keys=[user_id])

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_email": self.user.email if self.user else "Anonymous / System",
            "action": self.action,
            "resource": self.resource,
            "ip_address": self.ip_address,
            "details": self.details,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
