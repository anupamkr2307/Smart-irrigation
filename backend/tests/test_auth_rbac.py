import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.config import TestingConfig
from app.extensions import db
from app.models.user import User

@pytest.fixture
def client():
    app = create_app(TestingConfig)
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client

def test_seeded_user_login(client):
    # Test logging in with seeded farmer account
    res = client.post("/api/v1/auth/login", json={
        "email": "farmer@smartirrigation.org",
        "password": "Farmer@123"
    })
    assert res.status_code == 200
    data = res.get_json()
    assert "access_token" in data
    assert data["user"]["role"] == "farmer"

def test_invalid_login(client):
    res = client.post("/api/v1/auth/login", json={
        "email": "farmer@smartirrigation.org",
        "password": "WrongPassword"
    })
    assert res.status_code == 401

def test_user_registration(client):
    res = client.post("/api/v1/auth/register", json={
        "email": "new_farmer@test.org",
        "password": "Password@123",
        "full_name": "Kisan Bhai",
        "role": "farmer",
        "district": "Pune",
        "state": "Maharashtra"
    })
    assert res.status_code == 201
    data = res.get_json()
    assert data["user"]["email"] == "new_farmer@test.org"
    assert data["user"]["role"] == "farmer"

def test_admin_rbac_protection(client):
    # Log in as farmer
    login_res = client.post("/api/v1/auth/login", json={
        "email": "farmer@smartirrigation.org",
        "password": "Farmer@123"
    })
    token = login_res.get_json()["access_token"]

    # Attempt to access admin-only endpoint as farmer
    admin_res = client.get("/api/v1/admin/analytics", headers={
        "Authorization": f"Bearer {token}"
    })
    assert admin_res.status_code == 403

def test_admin_access(client):
    # Log in as admin
    login_res = client.post("/api/v1/auth/login", json={
        "email": "admin@smartirrigation.org",
        "password": "Admin@123"
    })
    token = login_res.get_json()["access_token"]

    admin_res = client.get("/api/v1/admin/analytics", headers={
        "Authorization": f"Bearer {token}"
    })
    assert admin_res.status_code == 200
    data = admin_res.get_json()
    assert "users" in data
    assert "operations" in data
