import os
import sys

# Ensure backend root is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.models.user import User
from app.models.marketplace import Product

app = create_app()
with app.app_context():
    users = User.query.all()
    products = Product.query.all()
    print("Flask App loaded successfully!")
    print(f"Total seeded users: {len(users)}")
    for u in users:
        print(f" - Role: {u.role:<12} Email: {u.email:<30} Name: {u.full_name}")
    print(f"Total seeded products: {len(products)}")
