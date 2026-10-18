import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.extensions import db
from app.models.marketplace import (
    ProductCategory, Product, Cart, CartItem, Order, OrderItem,
    QuotationRequest, Quotation, Review
)
from app.models.user import User
from app.models.system import AuditLog

marketplace_bp = Blueprint("marketplace", __name__, url_prefix="/api/v1/marketplace")

@marketplace_bp.route("/categories", methods=["GET"])
def get_categories():
    categories = ProductCategory.query.all()
    return jsonify({"categories": [c.to_dict() for c in categories]}), 200

@marketplace_bp.route("/products", methods=["GET"])
def get_products():
    category_slug = request.args.get("category")
    search = request.args.get("search", "").strip()
    retailer_id = request.args.get("retailer_id")

    query = Product.query.filter_by(is_active=True, is_approved=True)

    if category_slug:
        cat = ProductCategory.query.filter_by(slug=category_slug).first()
        if cat:
            query = query.filter_by(category_id=cat.id)

    if retailer_id:
        query = query.filter_by(retailer_id=int(retailer_id))

    if search:
        query = query.filter(Product.name.ilike(f"%{search}%") | Product.description.ilike(f"%{search}%"))

    products = query.order_by(Product.created_at.desc()).all()
    return jsonify({"products": [p.to_dict() for p in products]}), 200

@marketplace_bp.route("/products", methods=["POST"])
@jwt_required()
def create_product():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    if not user or user.role not in ["retailer", "admin"]:
        return jsonify({"error": "Only registered retailers or administrators can list products."}), 403

    data = request.get_json() or {}
    name = data.get("name", "").strip()
    price = float(data.get("price", 0.0))
    category_id = int(data.get("category_id", 1))
    stock_quantity = int(data.get("stock_quantity", 0))

    if not name or price <= 0:
        return jsonify({"error": "Product name and positive price are required."}), 400

    product = Product(
        retailer_id=user_id,
        category_id=category_id,
        name=name,
        description=data.get("description", ""),
        price=price,
        unit=data.get("unit", "piece"),
        stock_quantity=stock_quantity,
        image_url=data.get("image_url", "https://images.unsplash.com/photo-1592417817098-8f3d6eb228cc?auto=format&fit=crop&w=600&q=80"),
        is_approved=True,
        is_active=True
    )
    db.session.add(product)
    db.session.commit()

    return jsonify({"message": "Product created successfully.", "product": product.to_dict()}), 201

@marketplace_bp.route("/products/<int:product_id>", methods=["GET"])
def get_product_details(product_id):
    product = Product.query.get_or_404(product_id)
    return jsonify({"product": product.to_dict()}), 200

# Cart Endpoints
@marketplace_bp.route("/cart", methods=["GET"])
@jwt_required()
def get_cart():
    user_id = int(get_jwt_identity())
    cart = Cart.query.filter_by(user_id=user_id).first()
    if not cart:
        cart = Cart(user_id=user_id)
        db.session.add(cart)
        db.session.commit()
    return jsonify({"cart": cart.to_dict()}), 200

@marketplace_bp.route("/cart/items", methods=["POST"])
@jwt_required()
def add_to_cart():
    user_id = int(get_jwt_identity())
    data = request.get_json() or {}
    product_id = int(data.get("product_id", 0))
    quantity = int(data.get("quantity", 1))

    product = Product.query.get_or_404(product_id)
    if product.stock_quantity < quantity:
        return jsonify({"error": f"Insufficient stock. Only {product.stock_quantity} available."}), 400

    cart = Cart.query.filter_by(user_id=user_id).first()
    if not cart:
        cart = Cart(user_id=user_id)
        db.session.add(cart)
        db.session.flush()

    item = CartItem.query.filter_by(cart_id=cart.id, product_id=product_id).first()
    if item:
        item.quantity += quantity
    else:
        item = CartItem(cart_id=cart.id, product_id=product_id, quantity=quantity)
        db.session.add(item)

    db.session.commit()
    return jsonify({"message": "Item added to cart.", "cart": cart.to_dict()}), 200

@marketplace_bp.route("/cart/items/<int:item_id>", methods=["DELETE"])
@jwt_required()
def remove_from_cart(item_id):
    user_id = int(get_jwt_identity())
    item = CartItem.query.get_or_404(item_id)
    if item.cart.user_id != user_id:
        return jsonify({"error": "Unauthorized."}), 403

    cart = item.cart
    db.session.delete(item)
    db.session.commit()
    return jsonify({"message": "Item removed from cart.", "cart": cart.to_dict()}), 200

# Checkout & Orders
@marketplace_bp.route("/checkout", methods=["POST"])
@jwt_required()
def checkout():
    user_id = int(get_jwt_identity())
    cart = Cart.query.filter_by(user_id=user_id).first()
    if not cart or not cart.items:
        return jsonify({"error": "Cart is empty."}), 400

    data = request.get_json() or {}
    shipping = data.get("shipping", {})
    name = shipping.get("name", "").strip()
    phone = shipping.get("phone", "").strip()
    address = shipping.get("address", "").strip()
    city = shipping.get("city", "").strip()
    pincode = shipping.get("pincode", "").strip()
    payment_method = data.get("payment_method", "Mock UPI (Test Mode)")

    if not name or not phone or not address or not city or not pincode:
        return jsonify({"error": "All shipping fields are required."}), 400

    # Execute inside DB transaction for atomic stock decrement
    try:
        # Validate stock for all items first
        for item in cart.items:
            if item.product.stock_quantity < item.quantity:
                db.session.rollback()
                return jsonify({
                    "error": f"Product '{item.product.name}' has only {item.product.stock_quantity} in stock, but {item.quantity} requested."
                }), 400

        total_amount = sum(item.product.price * item.quantity for item in cart.items)
        first_product = cart.items[0].product
        retailer_id = first_product.retailer_id if first_product else None

        order_num = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        order = Order(
            order_number=order_num,
            customer_id=user_id,
            retailer_id=retailer_id,
            total_amount=round(total_amount, 2),
            currency="INR",
            status="confirmed",
            payment_status="mock_paid",
            payment_method=payment_method,
            shipping_name=name,
            shipping_phone=phone,
            shipping_address=address,
            shipping_city=city,
            shipping_pincode=pincode
        )
        db.session.add(order)
        db.session.flush()

        # Create order items and decrement product stock
        for item in cart.items:
            order_item = OrderItem(
                order_id=order.id,
                product_id=item.product.id,
                product_name=item.product.name,
                unit_price=item.product.price,
                quantity=item.quantity,
                subtotal=round(item.product.price * item.quantity, 2)
            )
            db.session.add(order_item)
            # Atomically decrement stock
            item.product.stock_quantity -= item.quantity

        # Clear cart items
        for item in cart.items:
            db.session.delete(item)

        # Audit log
        log = AuditLog(
            user_id=user_id,
            action="CREATE_ORDER",
            resource="Order",
            ip_address=request.remote_addr,
            details=f"Order {order_num} created for INR {total_amount} via {payment_method}"
        )
        db.session.add(log)
        db.session.commit()

        return jsonify({
            "message": "Order placed successfully! (Test Payment verified)",
            "order": order.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to complete checkout: {str(e)}"}), 500

@marketplace_bp.route("/orders", methods=["GET"])
@jwt_required()
def list_orders():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    if user.role == "retailer":
        orders = Order.query.filter_by(retailer_id=user_id).order_by(Order.created_at.desc()).all()
    elif user.role == "admin":
        orders = Order.query.order_by(Order.created_at.desc()).all()
    else:
        orders = Order.query.filter_by(customer_id=user_id).order_by(Order.created_at.desc()).all()

    return jsonify({"orders": [o.to_dict() for o in orders]}), 200

@marketplace_bp.route("/orders/<int:order_id>/status", methods=["PATCH"])
@jwt_required()
def update_order_status(order_id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    order = Order.query.get_or_404(order_id)

    if user.role not in ["retailer", "admin"] and order.retailer_id != user_id:
        return jsonify({"error": "Unauthorized."}), 403

    data = request.get_json() or {}
    new_status = data.get("status")
    if new_status not in ["pending", "confirmed", "shipped", "delivered", "cancelled"]:
        return jsonify({"error": "Invalid order status."}), 400

    order.status = new_status
    db.session.commit()
    return jsonify({"message": f"Order status updated to {new_status}.", "order": order.to_dict()}), 200

# Quotations
@marketplace_bp.route("/quotations", methods=["GET"])
@jwt_required()
def list_quotations():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    if user.role == "farmer":
        requests = QuotationRequest.query.filter_by(farmer_id=user_id).all()
    elif user.role == "retailer":
        requests = QuotationRequest.query.filter((QuotationRequest.retailer_id == user_id) | (QuotationRequest.retailer_id == None)).all()
    else:
        requests = QuotationRequest.query.all()

    return jsonify({"quotation_requests": [r.to_dict() for r in requests]}), 200

@marketplace_bp.route("/quotations", methods=["POST"])
@jwt_required()
def request_quotation():
    user_id = int(get_jwt_identity())
    data = request.get_json() or {}

    item_title = data.get("item_title", "").strip()
    quantity = data.get("quantity", "").strip()
    farm_location = data.get("farm_location", "").strip()
    description = data.get("description", "").strip()

    if not item_title or not quantity or not farm_location:
        return jsonify({"error": "Item title, quantity, and location are required."}), 400

    req = QuotationRequest(
        farmer_id=user_id,
        retailer_id=data.get("retailer_id"),
        item_title=item_title,
        quantity=quantity,
        farm_location=farm_location,
        description=description,
        status="pending"
    )
    db.session.add(req)
    db.session.commit()

    return jsonify({"message": "Quotation request posted.", "quotation_request": req.to_dict()}), 201

@marketplace_bp.route("/quotations/<int:request_id>/respond", methods=["POST"])
@jwt_required()
def respond_quotation(request_id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    if user.role not in ["retailer", "admin"]:
        return jsonify({"error": "Only retailers can submit price quotations."}), 403

    req_obj = QuotationRequest.query.get_or_404(request_id)
    data = request.get_json() or {}

    quote = Quotation(
        request_id=req_obj.id,
        retailer_id=user_id,
        offered_price=float(data.get("offered_price", 0.0)),
        estimated_delivery_days=int(data.get("estimated_delivery_days", 4)),
        remarks=data.get("remarks", "")
    )
    req_obj.status = "quoted"
    db.session.add(quote)
    db.session.commit()

    return jsonify({"message": "Quotation submitted.", "quotation": quote.to_dict()}), 201
