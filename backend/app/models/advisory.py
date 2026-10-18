from datetime import datetime
from app.extensions import db

class AgriculturalAdvisory(db.Model):
    __tablename__ = "agricultural_advisories"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), nullable=False)
    category = db.Column(db.String(60), default="Irrigation") # Irrigation, Soil Management, Pest/Disease, Water Conservation
    target_crop = db.Column(db.String(80), nullable=True) # None = general
    target_soil = db.Column(db.String(40), nullable=True)
    target_growth_stage = db.Column(db.String(40), nullable=True)
    advisory_text = db.Column(db.Text, nullable=False)
    precautions = db.Column(db.Text, nullable=True)
    is_emergency = db.Column(db.Boolean, default=False)
    is_specialist_verified = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "target_crop": self.target_crop,
            "target_soil": self.target_soil,
            "target_growth_stage": self.target_growth_stage,
            "advisory_text": self.advisory_text,
            "precautions": self.precautions,
            "is_emergency": self.is_emergency,
            "is_specialist_verified": self.is_specialist_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class Consultation(db.Model):
    __tablename__ = "consultations"

    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    specialist_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    
    subject = db.Column(db.String(200), nullable=False)
    issue_category = db.Column(db.String(60), nullable=False) # Pest & Disease, Soil & Nutrition, Irrigation Scheduling, Weather Damage, General Advisory
    crop_name = db.Column(db.String(80), nullable=True)
    soil_type = db.Column(db.String(40), nullable=True)
    location = db.Column(db.String(120), nullable=True)
    description = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(255), nullable=True)
    
    status = db.Column(db.String(30), default="open") # open, assigned, in_progress, resolved, closed
    rating = db.Column(db.Integer, nullable=True) # 1 to 5 stars
    feedback = db.Column(db.Text, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages = db.relationship("ConsultationMessage", backref="consultation", cascade="all, delete-orphan", order_by="ConsultationMessage.created_at.asc()")

    def to_dict(self):
        return {
            "id": self.id,
            "farmer_id": self.farmer_id,
            "farmer_name": self.farmer.full_name if self.farmer else "Unknown",
            "specialist_id": self.specialist_id,
            "specialist_name": self.specialist.full_name if self.specialist else None,
            "specialist_verified": self.specialist.specialist_profile.is_verified if (self.specialist and self.specialist.specialist_profile) else False,
            "subject": self.subject,
            "issue_category": self.issue_category,
            "crop_name": self.crop_name,
            "soil_type": self.soil_type,
            "location": self.location,
            "description": self.description,
            "image_url": self.image_url,
            "status": self.status,
            "rating": self.rating,
            "feedback": self.feedback,
            "message_count": len(self.messages),
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class ConsultationMessage(db.Model):
    __tablename__ = "consultation_messages"

    id = db.Column(db.Integer, primary_key=True)
    consultation_id = db.Column(db.Integer, db.ForeignKey("consultations.id"), nullable=False, index=True)
    sender_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    message = db.Column(db.Text, nullable=False)
    attachment_url = db.Column(db.String(255), nullable=True)
    is_specialist_advice = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    sender = db.relationship("User", foreign_keys=[sender_id])

    def to_dict(self):
        return {
            "id": self.id,
            "consultation_id": self.consultation_id,
            "sender_id": self.sender_id,
            "sender_name": self.sender.full_name if self.sender else "User",
            "sender_role": self.sender.role if self.sender else "farmer",
            "message": self.message,
            "attachment_url": self.attachment_url,
            "is_specialist_advice": self.is_specialist_advice,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
