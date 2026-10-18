from app.routes.auth import auth_bp
from app.routes.farms import farms_bp
from app.routes.weather import weather_bp
from app.routes.irrigation import irrigation_bp
from app.routes.crops import crops_bp
from app.routes.advisories import advisories_bp
from app.routes.consultations import consultations_bp
from app.routes.marketplace import marketplace_bp
from app.routes.admin import admin_bp
from app.routes.docs import docs_bp

__all__ = [
    "auth_bp", "farms_bp", "weather_bp", "irrigation_bp", "crops_bp",
    "advisories_bp", "consultations_bp", "marketplace_bp", "admin_bp", "docs_bp"
]
