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

def test_login_with_target_role_permissions(client):
    # Farmer logging in without portal restriction
    res1 = client.post("/api/v1/auth/login", json={
        "email": "farmer@smartirrigation.org",
        "password": "Farmer@123"
    })
    assert res1.status_code == 200
    assert "permissions" in res1.get_json()["user"]
    assert "manage_farms" in res1.get_json()["user"]["permissions"]

    # Farmer logging in with matching farmer portal
    res2 = client.post("/api/v1/auth/login", json={
        "email": "farmer@smartirrigation.org",
        "password": "Farmer@123",
        "target_role": "farmer"
    })
    assert res2.status_code == 200

    # Farmer attempting to log into retailer portal -> must be 403 Forbidden
    res3 = client.post("/api/v1/auth/login", json={
        "email": "farmer@smartirrigation.org",
        "password": "Farmer@123",
        "target_role": "retailer"
    })
    assert res3.status_code == 403
    assert "Permission Denied" in res3.get_json()["error"]
    assert res3.get_json()["user_role"] == "farmer"

    # Admin logging into retailer portal -> allowed
    res4 = client.post("/api/v1/auth/login", json={
        "email": "admin@smartirrigation.org",
        "password": "Admin@123",
        "target_role": "retailer"
    })
    assert res4.status_code == 200

def test_roles_metadata_endpoint(client):
    res = client.get("/api/v1/auth/roles")
    assert res.status_code == 200
    data = res.get_json()
    assert "roles" in data
    assert "farmer" in data["roles"]
    assert "retailer" in data["roles"]
    assert "specialist" in data["roles"]
    assert "customer" in data["roles"]
    assert "admin" in data["roles"]
    assert len(data["roles"]["farmer"]["permissions"]) > 0

def test_specialist_and_retailer_registration(client):
    # Register specialist
    spec_res = client.post("/api/v1/auth/register", json={
        "email": "dr_patil@agriexpert.org",
        "password": "Password@123",
        "full_name": "Dr. Sunita Patil",
        "role": "specialist",
        "specialization": "Plant Pathology",
        "qualification": "Ph.D. Agriculture",
        "years_experience": 8,
        "organization": "ICAR Central Institute"
    })
    assert spec_res.status_code == 201
    spec_data = spec_res.get_json()
    assert spec_data["user"]["role"] == "specialist"
    assert spec_data["user"]["profile"]["specialization"] == "Plant Pathology"
    assert "publish_advisories" in spec_data["permissions"]

    # Register retailer
    ret_res = client.post("/api/v1/auth/register", json={
        "email": "green_inputs@retail.org",
        "password": "Password@123",
        "full_name": "Green Earth Inputs",
        "role": "retailer",
        "shop_name": "Green Earth Bio-Agri",
        "gst_number": "27AAAAA0000A1Z5",
        "city": "Nashik",
        "state": "Maharashtra"
    })
    assert ret_res.status_code == 201
    ret_data = ret_res.get_json()
    assert ret_data["user"]["role"] == "retailer"
    assert ret_data["user"]["profile"]["shop_name"] == "Green Earth Bio-Agri"
    assert "manage_inventory" in ret_data["permissions"]

