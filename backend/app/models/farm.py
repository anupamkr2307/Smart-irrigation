from datetime import datetime
from app.extensions import db

class Farm(db.Model):
    __tablename__ = "farms"

    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    total_area = db.Column(db.Float, nullable=False, default=1.0)
    area_unit = db.Column(db.String(20), default="acres") # acres or hectares
    location_name = db.Column(db.String(120), nullable=False)
    state = db.Column(db.String(80), nullable=True)
    district = db.Column(db.String(80), nullable=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    plots = db.relationship("FarmPlot", backref="farm", cascade="all, delete-orphan", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "farmer_id": self.farmer_id,
            "name": self.name,
            "total_area": self.total_area,
            "area_unit": self.area_unit,
            "location_name": self.location_name,
            "state": self.state,
            "district": self.district,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "plot_count": len(self.plots),
            "plots": [p.to_dict() for p in self.plots],
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class FarmPlot(db.Model):
    __tablename__ = "farm_plots"

    id = db.Column(db.Integer, primary_key=True)
    farm_id = db.Column(db.Integer, db.ForeignKey("farms.id"), nullable=False, index=True)
    plot_name = db.Column(db.String(80), nullable=False)
    area = db.Column(db.Float, nullable=False, default=1.0)
    soil_type = db.Column(db.String(40), nullable=False, default="loamy") # sandy, loamy, clay, silty, sandy loam, clay loam
    current_crop = db.Column(db.String(80), nullable=True)
    growth_stage = db.Column(db.String(40), nullable=True, default="Mid-Season")
    irrigation_system = db.Column(db.String(40), default="Drip") # Drip, Sprinkler, Furrow, Basin
    soil_ph = db.Column(db.Float, default=6.8)
    target_moisture_pct = db.Column(db.Float, default=25.0)

    # Relationships
    soil_records = db.relationship("SoilRecord", backref="plot", cascade="all, delete-orphan", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "farm_id": self.farm_id,
            "plot_name": self.plot_name,
            "area": self.area,
            "soil_type": self.soil_type,
            "current_crop": self.current_crop,
            "growth_stage": self.growth_stage,
            "irrigation_system": self.irrigation_system,
            "soil_ph": self.soil_ph,
            "target_moisture_pct": self.target_moisture_pct
        }

class SoilRecord(db.Model):
    __tablename__ = "soil_records"

    id = db.Column(db.Integer, primary_key=True)
    plot_id = db.Column(db.Integer, db.ForeignKey("farm_plots.id"), nullable=False)
    sampled_at = db.Column(db.DateTime, default=datetime.utcnow)
    soil_type = db.Column(db.String(40), nullable=False)
    ph = db.Column(db.Float, nullable=False)
    moisture_pct = db.Column(db.Float, nullable=False)
    organic_matter_pct = db.Column(db.Float, nullable=True)
    nitrogen_ppm = db.Column(db.Float, nullable=True)
    phosphorus_ppm = db.Column(db.Float, nullable=True)
    potassium_ppm = db.Column(db.Float, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "plot_id": self.plot_id,
            "sampled_at": self.sampled_at.isoformat() if self.sampled_at else None,
            "soil_type": self.soil_type,
            "ph": self.ph,
            "moisture_pct": self.moisture_pct,
            "organic_matter_pct": self.organic_matter_pct,
            "nitrogen_ppm": self.nitrogen_ppm,
            "phosphorus_ppm": self.phosphorus_ppm,
            "potassium_ppm": self.potassium_ppm
        }

class WeatherRecord(db.Model):
    __tablename__ = "weather_records"

    id = db.Column(db.Integer, primary_key=True)
    location_name = db.Column(db.String(120), nullable=False, index=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    temperature_c = db.Column(db.Float, nullable=False)
    relative_humidity_pct = db.Column(db.Float, nullable=False)
    rainfall_recent_mm = db.Column(db.Float, default=0.0)
    rainfall_forecast_mm = db.Column(db.Float, default=0.0)
    wind_speed_ms = db.Column(db.Float, default=2.0)
    solar_radiation_mj = db.Column(db.Float, default=15.0)
    et0_mm = db.Column(db.Float, nullable=False)
    source = db.Column(db.String(60), default="Open-Meteo")

    def to_dict(self):
        return {
            "id": self.id,
            "location_name": self.location_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "temperature_c": self.temperature_c,
            "relative_humidity_pct": self.relative_humidity_pct,
            "rainfall_recent_mm": self.rainfall_recent_mm,
            "rainfall_forecast_mm": self.rainfall_forecast_mm,
            "wind_speed_ms": self.wind_speed_ms,
            "solar_radiation_mj": self.solar_radiation_mj,
            "et0_mm": self.et0_mm,
            "source": self.source
        }
