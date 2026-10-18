import os
from datetime import timedelta

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "smart-irrigation-super-secret-key-2026")
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "jwt-agriculture-key-2026-production")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=7)
    
    # SQLite default, PostgreSQL compatible
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'smart_irrigation.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload folder
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    
    # ML model path
    ML_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "ml", "models", "irrigation_model.joblib"))
    ML_METRICS_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "ml", "models", "metrics.json"))
    DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset", "irrigation_data.csv"))

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
