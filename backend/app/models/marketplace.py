from datetime import datetime
from app.extensions import db

class ProductCategory(db.Model):
    __tablename__ = "product_categories"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    slug = db.Column(db.String(80), unique=True, nullable=False)
    description = db.Column(db.String(255), nullable=True)

    products = db.relationship("Product", backref="category", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "product_count": len(self.products)
        }

class Product(db.Model):
    __tablename__ = "products"

    id = db.Column(db.Integer, primary_key=True)
    retailer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    category_id = db.Column(db.Integer, db.ForeignKey("product_categories.id"), nullable=False, index=True)
    
    name = db.Column(db.String(150), nullable=False)
    description = db.Column(db.Text, nullable=False)
    price = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(30), default="piece") # piece, kg, liter, meter, bag
    stock_quantity = db.Column(db.Integer, nullable=False, default=0)
    image_url = db.Column(db.String(255), nullable=True)
    is_approved = db.Column(db.Boolean, default=True) # moderation
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    retailer = db.relationship("User", foreign_keys=[retailer_id])
    reviews = db.relationship("Review", backref="product", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        avg_rating = 0.0
        if self.reviews:
            avg_rating = sum(r.rating for r in self.reviews) / len(self.reviews)

        return {
            "id": self.id,
            "retailer_id": self.retailer_id,
            "retailer_name": self.retailer.full_name if self.retailer else "Retailer",
            "shop_name": self.retailer.retailer_profile.shop_name if (self.retailer and self.retailer.retailer_profile) else None,
            "category_id": self.category_id,
            "category_name": self.category.name if self.category else "General",
            "name": self.name,
            "description": self.description,
            "price": self.price,
            "unit": self.unit,
            "stock_quantity": self.stock_quantity,
            "image_url": self.image_url,
            "is_approved": self.is_approved,
            "is_active": self.is_active,
            "rating": round(avg_rating, 1),
            "review_count": len(self.reviews),
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class Cart(db.Model):
    __tablename__ = "carts"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    items = db.relationship("CartItem", backref="cart", cascade="all, delete-orphan", lazy=True)

    def to_dict(self):
        total = sum(item.product.price * item.quantity for item in self.items if item.product)
        return {
            "id": self.id,
            "user_id": self.user_id,
            "items": [item.to_dict() for item in self.items],
            "total_items": sum(item.quantity for item in self.items),
            "total_amount": round(total, 2)
        }

class CartItem(db.Model):
    __tablename__ = "cart_items"

    id = db.Column(db.Integer, primary_key=True)
    cart_id = db.Column(db.Integer, db.ForeignKey("carts.id"), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id"), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=1)

    product = db.relationship("Product")

    def to_dict(self):
        return {
            "id": self.id,
            "product_id": self.product_id,
            "product_name": self.product.name if self.product else "Deleted Product",
            "price": self.product.price if self.product else 0.0,
            "unit": self.product.unit if self.product else "",
            "image_url": self.product.image_url if self.product else None,
            "available_stock": self.product.stock_quantity if self.product else 0,
            "quantity": self.quantity,
            "subtotal": round((self.product.price if self.product else 0.0) * self.quantity, 2)
        }

class Order(db.Model):
    __tablename__ = "orders"

    id = db.Column(db.Integer, primary_key=True)
    order_number = db.Column(db.String(32), unique=True, nullable=False, index=True)
    customer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    retailer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    
    total_amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(10), default="INR")
    status = db.Column(db.String(30), default="pending") # pending, confirmed, shipped, delivered, cancelled
    payment_status = db.Column(db.String(30), default="mock_paid") # mock_paid, cod, pending
    payment_method = db.Column(db.String(40), default="Mock UPI Payment")
    
    shipping_name = db.Column(db.String(100), nullable=False)
    shipping_phone = db.Column(db.String(20), nullable=False)
    shipping_address = db.Column(db.Text, nullable=False)
    shipping_city = db.Column(db.String(60), nullable=False)
    shipping_pincode = db.Column(db.String(20), nullable=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    items = db.relationship("OrderItem", backref="order", cascade="all, delete-orphan", lazy=True)
    retailer = db.relationship("User", foreign_keys=[retailer_id], overlaps="retailer_orders")

    def to_dict(self):
        return {
            "id": self.id,
            "order_number": self.order_number,
            "customer_id": self.customer_id,
            "customer_name": self.customer.full_name if self.customer else "Customer",
            "retailer_id": self.retailer_id,
            "retailer_name": self.retailer.full_name if self.retailer else "Direct Vendor",
            "total_amount": self.total_amount,
            "currency": self.currency,
            "status": self.status,
            "payment_status": self.payment_status,
            "payment_method": self.payment_method,
            "shipping": {
                "name": self.shipping_name,
                "phone": self.shipping_phone,
                "address": self.shipping_address,
                "city": self.shipping_city,
                "pincode": self.shipping_pincode
            },
            "items": [item.to_dict() for item in self.items],
            "item_count": len(self.items),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class OrderItem(db.Model):
    __tablename__ = "order_items"

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey("orders.id"), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id"), nullable=False)
    product_name = db.Column(db.String(150), nullable=False)
    unit_price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    subtotal = db.Column(db.Float, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "product_id": self.product_id,
            "product_name": self.product_name,
            "unit_price": self.unit_price,
            "quantity": self.quantity,
            "subtotal": self.subtotal
        }

class QuotationRequest(db.Model):
    __tablename__ = "quotation_requests"

    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    retailer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    item_title = db.Column(db.String(150), nullable=False)
    quantity = db.Column(db.String(50), nullable=False)
    farm_location = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(30), default="pending") # pending, quoted, accepted, rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    farmer = db.relationship("User", foreign_keys=[farmer_id])
    retailer = db.relationship("User", foreign_keys=[retailer_id])
    quotations = db.relationship("Quotation", backref="request", cascade="all, delete-orphan", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "farmer_id": self.farmer_id,
            "farmer_name": self.farmer.full_name if self.farmer else "Farmer",
            "retailer_id": self.retailer_id,
            "retailer_name": self.retailer.full_name if self.retailer else "All Retailers",
            "item_title": self.item_title,
            "quantity": self.quantity,
            "farm_location": self.farm_location,
            "description": self.description,
            "status": self.status,
            "quotations": [q.to_dict() for q in self.quotations],
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class Quotation(db.Model):
    __tablename__ = "quotations"

    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.Integer, db.ForeignKey("quotation_requests.id"), nullable=False)
    retailer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    offered_price = db.Column(db.Float, nullable=False)
    estimated_delivery_days = db.Column(db.Integer, default=5)
    valid_until = db.Column(db.DateTime, nullable=True)
    remarks = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(30), default="active") # active, accepted, rejected, expired
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    retailer = db.relationship("User", foreign_keys=[retailer_id])

    def to_dict(self):
        return {
            "id": self.id,
            "request_id": self.request_id,
            "retailer_id": self.retailer_id,
            "retailer_name": self.retailer.full_name if self.retailer else "Retailer",
            "shop_name": self.retailer.retailer_profile.shop_name if (self.retailer and self.retailer.retailer_profile) else None,
            "offered_price": self.offered_price,
            "estimated_delivery_days": self.estimated_delivery_days,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "remarks": self.remarks,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class Review(db.Model):
    __tablename__ = "reviews"

    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    rating = db.Column(db.Integer, nullable=False) # 1 - 5
    comment = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User")

    def to_dict(self):
        return {
            "id": self.id,
            "product_id": self.product_id,
            "user_id": self.user_id,
            "user_name": self.user.full_name if self.user else "Customer",
            "rating": self.rating,
            "comment": self.comment,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
