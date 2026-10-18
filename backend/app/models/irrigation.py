from datetime import datetime
from app.extensions import db

class CropReference(db.Model):
    __tablename__ = "crop_references"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    scientific_name = db.Column(db.String(120), nullable=True)
    category = db.Column(db.String(60), default="Cereal") # Cereal, Pulse, Cash Crop, Vegetable, Oilseed
    season = db.Column(db.String(60), default="Kharif / Rabi")
    kc_initial = db.Column(db.Float, default=0.40)
    kc_development = db.Column(db.Float, default=0.75)
    kc_mid_season = db.Column(db.Float, default=1.15)
    kc_late_season = db.Column(db.Float, default=0.55)
    opt_temp_min = db.Column(db.Float, default=15.0)
    opt_temp_max = db.Column(db.Float, default=32.0)
    opt_ph_min = db.Column(db.Float, default=6.0)
    opt_ph_max = db.Column(db.Float, default=7.5)
    water_demand_level = db.Column(db.String(30), default="Medium") # Low, Medium, High, Very High
    growth_duration_days = db.Column(db.Integer, default=120)
    root_depth_cm = db.Column(db.Integer, default=60)
    default_irrigation_method = db.Column(db.String(40), default="Drip")
    soil_preferences = db.Column(db.String(150), default="loamy, clay loam")
    fertilizer_guidelines = db.Column(db.Text, nullable=True)
    pest_disease_precautions = db.Column(db.Text, nullable=True)
    risks_limitations = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "scientific_name": self.scientific_name,
            "category": self.category,
            "season": self.season,
            "kc": {
                "Initial": self.kc_initial,
                "Development": self.kc_development,
                "Mid-Season": self.kc_mid_season,
                "Late Season": self.kc_late_season
            },
            "temperature_range": [self.opt_temp_min, self.opt_temp_max],
            "ph_range": [self.opt_ph_min, self.opt_ph_max],
            "water_demand_level": self.water_demand_level,
            "growth_duration_days": self.growth_duration_days,
            "root_depth_cm": self.root_depth_cm,
            "default_irrigation_method": self.default_irrigation_method,
            "soil_preferences": self.soil_preferences,
            "fertilizer_guidelines": self.fertilizer_guidelines,
            "pest_disease_precautions": self.pest_disease_precautions,
            "risks_limitations": self.risks_limitations
        }

class IrrigationPrediction(db.Model):
    __tablename__ = "irrigation_predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    farm_id = db.Column(db.Integer, db.ForeignKey("farms.id"), nullable=True)
    plot_id = db.Column(db.Integer, db.ForeignKey("farm_plots.id"), nullable=True)
    
    # Input snapshot
    location_name = db.Column(db.String(120), nullable=False)
    soil_type = db.Column(db.String(40), nullable=False)
    crop_type = db.Column(db.String(80), nullable=False)
    crop_growth_stage = db.Column(db.String(40), nullable=False)
    farm_area = db.Column(db.Float, nullable=False)
    area_unit = db.Column(db.String(20), default="acres")
    
    # Weather & Soil snapshot
    temperature_c = db.Column(db.Float, nullable=False)
    relative_humidity_pct = db.Column(db.Float, nullable=False)
    soil_moisture_pct = db.Column(db.Float, nullable=False)
    recent_rainfall_mm = db.Column(db.Float, default=0.0)
    forecast_rainfall_mm = db.Column(db.Float, default=0.0)
    wind_speed_ms = db.Column(db.Float, default=2.0)
    solar_radiation_mj = db.Column(db.Float, default=15.0)
    et0_mm = db.Column(db.Float, nullable=False)
    etc_mm = db.Column(db.Float, nullable=False)
    effective_rainfall_mm = db.Column(db.Float, default=0.0)

    # Prediction outputs
    predicted_net_mm = db.Column(db.Float, nullable=False)
    predicted_net_liters = db.Column(db.Float, nullable=False)
    gross_irrigation_mm = db.Column(db.Float, nullable=False)
    gross_irrigation_liters = db.Column(db.Float, nullable=False)
    irrigation_efficiency_pct = db.Column(db.Float, default=90.0)
    irrigation_frequency = db.Column(db.String(100), nullable=False)
    recommended_irrigation_method = db.Column(db.String(60), nullable=False)
    model_version = db.Column(db.String(60), default="1.0.0-LinearRegression")
    confidence_notes = db.Column(db.Text, nullable=True)
    water_saving_tips = db.Column(db.Text, nullable=True)
    soil_precautions = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "farm_id": self.farm_id,
            "plot_id": self.plot_id,
            "inputs": {
                "location_name": self.location_name,
                "soil_type": self.soil_type,
                "crop_type": self.crop_type,
                "crop_growth_stage": self.crop_growth_stage,
                "farm_area": self.farm_area,
                "area_unit": self.area_unit,
                "temperature_c": self.temperature_c,
                "relative_humidity_pct": self.relative_humidity_pct,
                "soil_moisture_pct": self.soil_moisture_pct,
                "recent_rainfall_mm": self.recent_rainfall_mm,
                "forecast_rainfall_mm": self.forecast_rainfall_mm,
                "et0_mm": self.et0_mm,
                "etc_mm": self.etc_mm,
                "effective_rainfall_mm": self.effective_rainfall_mm
            },
            "outputs": {
                "net_irrigation_mm": self.predicted_net_mm,
                "net_irrigation_liters": self.predicted_net_liters,
                "gross_irrigation_mm": self.gross_irrigation_mm,
                "gross_irrigation_liters": self.gross_irrigation_liters,
                "irrigation_efficiency_pct": self.irrigation_efficiency_pct,
                "irrigation_frequency": self.irrigation_frequency,
                "recommended_method": self.recommended_irrigation_method,
                "water_saving_tips": self.water_saving_tips,
                "soil_precautions": self.soil_precautions,
                "confidence_notes": self.confidence_notes,
                "model_version": self.model_version
            },
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class MLModelVersion(db.Model):
    __tablename__ = "ml_model_versions"

    id = db.Column(db.Integer, primary_key=True)
    version_tag = db.Column(db.String(40), unique=True, nullable=False)
    algorithm = db.Column(db.String(80), default="LinearRegression Pipeline")
    mae = db.Column(db.Float, nullable=False)
    rmse = db.Column(db.Float, nullable=False)
    r2_score = db.Column(db.Float, nullable=False)
    train_samples = db.Column(db.Integer, default=1200)
    test_samples = db.Column(db.Integer, default=300)
    file_path = db.Column(db.String(255), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "version_tag": self.version_tag,
            "algorithm": self.algorithm,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2_score": self.r2_score,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
