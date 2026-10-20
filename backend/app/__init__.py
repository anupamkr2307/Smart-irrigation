import os
from flask import Flask, jsonify, send_from_directory
from app.config import Config, DevelopmentConfig
from app.extensions import db, jwt, migrate, cors
from app.routes import (
    auth_bp, farms_bp, weather_bp, irrigation_bp, crops_bp,
    advisories_bp, consultations_bp, marketplace_bp, admin_bp, docs_bp
)
from app.utils.seed_data import seed_database

def create_app(config_class=DevelopmentConfig):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    jwt.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(farms_bp)
    app.register_blueprint(weather_bp)
    app.register_blueprint(irrigation_bp)
    app.register_blueprint(crops_bp)
    app.register_blueprint(advisories_bp)
    app.register_blueprint(consultations_bp)
    app.register_blueprint(marketplace_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(docs_bp)

    # Health check route
    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status": "healthy",
            "service": "Smart Irrigation Platform Backend",
            "version": "1.0.0"
        }), 200

    # Production static file serving and SPA catch-all
    dist_dir = app.config.get("FRONTEND_DIST_DIR")
    if dist_dir and os.path.exists(dist_dir):
        @app.route("/", defaults={"path": ""})
        @app.route("/<path:path>")
        def serve_frontend(path):
            if path.startswith("api/"):
                return jsonify({"error": "Endpoint not found"}), 404
            file_path = os.path.join(dist_dir, path)
            if path and os.path.isfile(file_path):
                return send_from_directory(dist_dir, path)
            return send_from_directory(dist_dir, "index.html")

    # Centralized Error Handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "Bad request", "details": str(error)}), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Resource not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return jsonify({"error": "Internal server error"}), 500

    # Ensure upload directory exists
    upload_dir = app.config.get("UPLOAD_FOLDER")
    if upload_dir and not os.path.exists(upload_dir):
        try:
            os.makedirs(upload_dir, exist_ok=True)
        except Exception:
            pass

    # Initialize tables and seed data automatically on startup
    with app.app_context():
        db.create_all()
        try:
            seed_database()
        except Exception as e:
            print(f"Database seed note: {e}")

    return app
