from flask import Blueprint, jsonify

docs_bp = Blueprint("docs", __name__, url_prefix="/api/v1")

@docs_bp.route("/docs", methods=["GET"])
def api_docs():
    swagger_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Smart Irrigation & Agriculture Lifecycle Management API",
            "version": "1.0.0",
            "description": "REST API documentation for the Smart Irrigation complete full-stack platform."
        },
        "servers": [{"url": "/api/v1"}],
        "paths": {
            "/auth/register": {"post": {"summary": "Register a new user (farmer, retailer, customer, specialist)"}},
            "/auth/login": {"post": {"summary": "Authenticate user and issue JWT access token"}},
            "/auth/me": {"get": {"summary": "Get authenticated user profile and details"}},
            "/farms": {
                "get": {"summary": "List user farms"},
                "post": {"summary": "Register a new farm with plots and geo-coordinates"}
            },
            "/weather/geocode": {"get": {"summary": "Search location coordinates by city/village name"}},
            "/weather/current": {"get": {"summary": "Fetch real-time weather and 7-day forecast via Open-Meteo"}},
            "/irrigation/predict": {"post": {"summary": "Run ML Linear Regression pipeline + FAO-56 water calculations"}},
            "/irrigation/history": {"get": {"summary": "Retrieve historical predictions and water requirements"}},
            "/crops": {"get": {"summary": "List reference agricultural crops"}},
            "/crops/recommend": {"post": {"summary": "Multi-criteria agronomic crop suitability recommendation"}},
            "/advisories": {"get": {"summary": "List tailored agricultural advisories"}},
            "/consultations": {
                "get": {"summary": "List consultations by role"},
                "post": {"summary": "Submit agricultural issue consultation"}
            },
            "/marketplace/products": {
                "get": {"summary": "Browse and search agricultural marketplace products"},
                "post": {"summary": "Retailer list new agricultural product"}
            },
            "/marketplace/checkout": {"post": {"summary": "Checkout cart items with atomic stock verification"}},
            "/marketplace/quotations": {"get": {"summary": "List quotation requests"}, "post": {"summary": "Post bulk quote request"}},
            "/admin/analytics": {"get": {"summary": "Platform operational metrics and water saved totals"}},
            "/admin/ml-metrics": {"get": {"summary": "Inspect MAE, RMSE, R² score, and feature weights"}}
        }
    }
    return jsonify(swagger_spec), 200
