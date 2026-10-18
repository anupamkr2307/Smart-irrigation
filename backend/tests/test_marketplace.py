import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.config import TestingConfig
from app.extensions import db
from app.models.marketplace import Product

@pytest.fixture
def client():
    app = create_app(TestingConfig)
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client

def test_browse_products(client):
    res = client.get("/api/v1/marketplace/products")
    assert res.status_code == 200
    data = res.get_json()
    assert "products" in data
    assert len(data["products"]) > 0

def test_cart_and_checkout_flow(client):
    # 1. Login as customer
    login_res = client.post("/api/v1/auth/login", json={
        "email": "customer@smartirrigation.org",
        "password": "Customer@123"
    })
    token = login_res.get_json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Get first product id
    prod_res = client.get("/api/v1/marketplace/products")
    product = prod_res.get_json()["products"][0]
    initial_stock = product["stock_quantity"]

    # 3. Add to cart
    add_res = client.post("/api/v1/marketplace/cart/items", json={
        "product_id": product["id"],
        "quantity": 2
    }, headers=headers)
    assert add_res.status_code == 200

    # 4. Checkout
    checkout_res = client.post("/api/v1/marketplace/checkout", json={
        "shipping": {
            "name": "Ananya Sharma",
            "phone": "9876543210",
            "address": "Green Meadows, Farmhouse 7",
            "city": "Nashik",
            "pincode": "422003"
        },
        "payment_method": "Mock UPI Test Mode"
    }, headers=headers)
    assert checkout_res.status_code == 201
    order_data = checkout_res.get_json()["order"]
    assert order_data["payment_status"] == "mock_paid"
    assert order_data["total_amount"] > 0

    # 5. Verify stock decreased by 2
    verify_res = client.get(f"/api/v1/marketplace/products/{product['id']}")
    updated_product = verify_res.get_json()["product"]
    assert updated_product["stock_quantity"] == initial_stock - 2
