"""
Database Seeder
Populates initial reference crops, product categories, initial marketplace products,
sample advisories, and pre-configured test users for all 5 roles.
"""

from app.extensions import db
from app.models.user import User, FarmerProfile, SpecialistProfile, RetailerProfile
from app.models.farm import Farm, FarmPlot
from app.models.irrigation import CropReference, MLModelVersion
from app.models.advisory import AgriculturalAdvisory
from app.models.marketplace import ProductCategory, Product
from app.services.crop_recommender import REFERENCE_CROPS

def seed_database():
    # 1. Seed Users for 5 roles if not present
    seed_users = [
        {
            "email": "farmer@smartirrigation.org",
            "password": "Farmer@123",
            "full_name": "Ramesh Patel",
            "phone": "+91 98765 43210",
            "role": "farmer",
            "profile_type": "farmer",
            "profile_data": {
                "farm_name": "Patel Green Agro Farms",
                "preferred_language": "English / Marathi",
                "district": "Nashik",
                "state": "Maharashtra",
                "primary_crops": "Grapes, Tomato, Wheat"
            }
        },
        {
            "email": "retailer@smartirrigation.org",
            "password": "Retailer@123",
            "full_name": "Suresh Agro Traders",
            "phone": "+91 98220 11223",
            "role": "retailer",
            "profile_type": "retailer",
            "profile_data": {
                "shop_name": "Kisan Seva Kendra & Agri Supplies",
                "gst_number": "27AABCS1429B1ZB",
                "license_number": "AGRI-RET-4921",
                "address": "Plot 42, Market Yard Road, Panchavati",
                "city": "Nashik",
                "state": "Maharashtra",
                "delivery_supported": True
            }
        },
        {
            "email": "customer@smartirrigation.org",
            "password": "Customer@123",
            "full_name": "Ananya Sharma",
            "phone": "+91 97654 32109",
            "role": "customer"
        },
        {
            "email": "specialist@smartirrigation.org",
            "password": "Specialist@123",
            "full_name": "Dr. Arvind Deshmukh",
            "phone": "+91 94231 88776",
            "role": "specialist",
            "profile_type": "specialist",
            "profile_data": {
                "specialization": "Irrigation Engineering & Soil Physics",
                "qualification": "Ph.D. in Water Resource Management, MPKV Rahuri",
                "years_experience": 14,
                "license_number": "AGRI-SPEC-8820",
                "organization": "Maharashtra Agricultural Research Council",
                "is_verified": True,
                "bio": "Specialist in micro-irrigation systems, fertigation scheduling, and water-stressed crop physiology."
            }
        },
        {
            "email": "admin@smartirrigation.org",
            "password": "Admin@123",
            "full_name": "System Administrator",
            "phone": "+91 90000 00001",
            "role": "admin"
        }
    ]

    for u_data in seed_users:
        existing = User.query.filter_by(email=u_data["email"]).first()
        if not existing:
            user = User(
                email=u_data["email"],
                full_name=u_data["full_name"],
                phone=u_data.get("phone"),
                role=u_data["role"]
            )
            user.set_password(u_data["password"])
            db.session.add(user)
            db.session.flush()

            # Attach role profile
            if u_data.get("profile_type") == "farmer":
                p = FarmerProfile(user_id=user.id, **u_data["profile_data"])
                db.session.add(p)
                # Add default farm & plot
                farm = Farm(
                    farmer_id=user.id,
                    name="Nashik Vineyard & Vegetable Plot",
                    total_area=4.5,
                    area_unit="acres",
                    location_name="Nashik",
                    state="Maharashtra",
                    district="Nashik",
                    latitude=19.9975,
                    longitude=73.7898
                )
                db.session.add(farm)
                db.session.flush()

                plot1 = FarmPlot(
                    farm_id=farm.id,
                    plot_name="North Plot - Vegetables",
                    area=2.0,
                    soil_type="loamy",
                    current_crop="Tomato",
                    growth_stage="Mid-Season",
                    irrigation_system="Drip",
                    soil_ph=6.8,
                    target_moisture_pct=26.0
                )
                plot2 = FarmPlot(
                    farm_id=farm.id,
                    plot_name="South Plot - Grains",
                    area=2.5,
                    soil_type="clay loam",
                    current_crop="Wheat",
                    growth_stage="Development",
                    irrigation_system="Sprinkler",
                    soil_ph=7.2,
                    target_moisture_pct=24.0
                )
                db.session.add_all([plot1, plot2])

            elif u_data.get("profile_type") == "specialist":
                p = SpecialistProfile(user_id=user.id, **u_data["profile_data"])
                db.session.add(p)
            elif u_data.get("profile_type") == "retailer":
                p = RetailerProfile(user_id=user.id, **u_data["profile_data"])
                db.session.add(p)

    # 2. Seed Crop References
    for c in REFERENCE_CROPS:
        existing_crop = CropReference.query.filter_by(name=c["name"]).first()
        if not existing_crop:
            crop_ref = CropReference(
                name=c["name"],
                scientific_name=c["scientific_name"],
                category=c["category"],
                season=c["season"],
                opt_temp_min=c["opt_temp"][0],
                opt_temp_max=c["opt_temp"][1],
                opt_ph_min=c["opt_ph"][0],
                opt_ph_max=c["opt_ph"][1],
                water_demand_level=c["water_demand"],
                growth_duration_days=c["duration_days"],
                root_depth_cm=c["root_depth_cm"],
                default_irrigation_method=c["default_method"],
                soil_preferences=", ".join(c["opt_soils"]),
                fertilizer_guidelines=c["fertilizer_tips"],
                pest_disease_precautions=c["pest_precautions"],
                risks_limitations=c["risks"]
            )
            db.session.add(crop_ref)

    # 3. Seed Product Categories
    categories = [
        {"name": "Irrigation Equipment", "slug": "irrigation-equipment", "description": "Drip kits, inline drippers, micro sprinklers, pumps, and water pipes"},
        {"name": "Seeds & Planting Materials", "slug": "seeds", "description": "Certified high-yielding and drought-resilient seed varieties"},
        {"name": "Fertilizers & Nutrients", "slug": "fertilizers", "description": "Organic compost, water-soluble NPK fertigation blends, micronutrients"},
        {"name": "Tools & Machinery", "slug": "tools-machinery", "description": "Soil moisture sensors, tensiometers, sprayers, and farm implements"},
        {"name": "Organic Crop Protection", "slug": "organic-inputs", "description": "Neem-based repellents, bio-fungicides, and soil conditioning microbes"}
    ]
    cat_objs = {}
    for cat_data in categories:
        existing_cat = ProductCategory.query.filter_by(slug=cat_data["slug"]).first()
        if not existing_cat:
            cat = ProductCategory(**cat_data)
            db.session.add(cat)
            db.session.flush()
            cat_objs[cat.slug] = cat
        else:
            cat_objs[existing_cat.slug] = existing_cat

    # 4. Seed Initial Products linked to Retailer
    retailer_user = User.query.filter_by(email="retailer@smartirrigation.org").first()
    if retailer_user:
        products = [
            {
                "retailer_id": retailer_user.id,
                "category_id": cat_objs["irrigation-equipment"].id,
                "name": "Precision Inline Drip Irrigation Kit (1 Acre)",
                "description": "Complete drip irrigation kit including 16mm inline drip lateral (400m), screen filter (20 m3/hr), venturi fertigation injector, connector fittings, and flush valves. Highly efficient for row vegetables and orchards.",
                "price": 14500.0,
                "unit": "set",
                "stock_quantity": 25,
                "image_url": "https://images.unsplash.com/photo-1592417817098-8f3d6eb228cc?auto=format&fit=crop&w=600&q=80"
            },
            {
                "retailer_id": retailer_user.id,
                "category_id": cat_objs["irrigation-equipment"].id,
                "name": "Micro-Sprinkler Overhead Kit (0.5 Acre)",
                "description": "360-degree rotating micro-sprinklers with support rods and flexible feeder tubes. Uniform precipitation ideal for onion, garlic, groundnut, and vegetable nurseries.",
                "price": 6800.0,
                "unit": "set",
                "stock_quantity": 40,
                "image_url": "https://images.unsplash.com/photo-1563514227147-6d2ff665a6a0?auto=format&fit=crop&w=600&q=80"
            },
            {
                "retailer_id": retailer_user.id,
                "category_id": cat_objs["tools-machinery"].id,
                "name": "Digital Volumetric Soil Moisture & Temp Sensor",
                "description": "Rugged direct-insertion probe measuring volumetric water content (0-100%) and soil temperature. Backlit LCD display with immediate agricultural trigger thresholds.",
                "price": 2850.0,
                "unit": "piece",
                "stock_quantity": 50,
                "image_url": "https://images.unsplash.com/photo-1581092160607-ee22621dd758?auto=format&fit=crop&w=600&q=80"
            },
            {
                "retailer_id": retailer_user.id,
                "category_id": cat_objs["fertilizers"].id,
                "name": "Water Soluble NPK 19-19-19 Fertigation Blend (25 kg)",
                "description": "100% water-soluble high-purity fertilizer engineered specifically for drip fertigation. Free of chlorine and insoluble residues.",
                "price": 2150.0,
                "unit": "bag",
                "stock_quantity": 80,
                "image_url": "https://images.unsplash.com/photo-1585314062340-f1a5a7c9328d?auto=format&fit=crop&w=600&q=80"
            },
            {
                "retailer_id": retailer_user.id,
                "category_id": cat_objs["organic-inputs"].id,
                "name": "Pure Cold-Pressed Neem Oil (10,000 PPM Azadirachtin) 1L",
                "description": "Bio-organic pest repellent and antifeedant against sucking pests, whiteflies, thrips, and aphids. Safe for beneficial pollinators.",
                "price": 650.0,
                "unit": "liter",
                "stock_quantity": 120,
                "image_url": "https://images.unsplash.com/photo-1615485290382-441e4d049cb5?auto=format&fit=crop&w=600&q=80"
            },
            {
                "retailer_id": retailer_user.id,
                "category_id": cat_objs["seeds"].id,
                "name": "Hybrid Drought-Tolerant Tomato Seeds (F1, 10g)",
                "description": "Indeterminate hybrid with deep rooting habit, resistance to TYLCV and bacterial wilt. High yield of uniform, firm 90-100g fruits.",
                "price": 890.0,
                "unit": "pack",
                "stock_quantity": 60,
                "image_url": "https://images.unsplash.com/photo-1592841200221-a6898f307baa?auto=format&fit=crop&w=600&q=80"
            }
        ]

        for p_data in products:
            existing_p = Product.query.filter_by(name=p_data["name"]).first()
            if not existing_p:
                prod = Product(**p_data)
                db.session.add(prod)

    # 5. Seed Agricultural Advisories
    advisories = [
        {
            "title": "Optimizing Drip Irrigation Intervals During High Temperature Spikes",
            "category": "Irrigation",
            "target_crop": None,
            "target_soil": None,
            "target_growth_stage": None,
            "advisory_text": "When ambient temperatures exceed 35°C, split your daily irrigation runtime into two equal pulses (e.g. 7:00 AM and 5:30 PM). This reduces midday evaporation loss, prevents surface crusting, and ensures water stays in the active root capillary zone.",
            "precautions": "Avoid irrigating between 11:30 AM and 3:30 PM as hot water in exposed black lateral pipes can scald tender feeder roots.",
            "is_emergency": False,
            "is_specialist_verified": True
        },
        {
            "title": "Soil Moisture Management in Heavy Clay and Black Cotton Soils",
            "category": "Soil Management",
            "target_crop": None,
            "target_soil": "clay",
            "target_growth_stage": None,
            "advisory_text": "Clay soils have high water-holding capacity (180 mm/m) but slow infiltration (4 mm/hr). Over-watering quickly induces anaerobic conditions and root rot. Irrigate only when volumetric soil moisture drops below 22%.",
            "precautions": "Do not apply heavy flood watering immediately after a long dry spell; deep cracking can cause uneven percolation.",
            "is_emergency": False,
            "is_specialist_verified": True
        },
        {
            "title": "Critical Water Stage Alert: Mid-Season Flowering & Fruit Set",
            "category": "Water Conservation",
            "target_crop": "Tomato",
            "target_soil": None,
            "target_growth_stage": "Mid-Season",
            "advisory_text": "Tomato plants under flowering and fruit setting are highly vulnerable to moisture oscillation. Water stress during this window results in flower drop and blossom end rot. Maintain steady soil moisture between 24% and 28%.",
            "precautions": "Apply calcium nitrate via drip fertigation to prevent physiological blossom end rot.",
            "is_emergency": True,
            "is_specialist_verified": True
        }
    ]

    for adv in advisories:
        existing_adv = AgriculturalAdvisory.query.filter_by(title=adv["title"]).first()
        if not existing_adv:
            advisory_obj = AgriculturalAdvisory(**adv)
            db.session.add(advisory_obj)

    # 6. Seed Model Version record
    existing_mv = MLModelVersion.query.filter_by(version_tag="v1.0-LinearRegression").first()
    if not existing_mv:
        mv = MLModelVersion(
            version_tag="v1.0-LinearRegression",
            algorithm="Scikit-Learn Pipeline (OneHotEncoder + StandardScaler + LinearRegression)",
            mae=1.7781,
            rmse=2.7571,
            r2_score=0.6982,
            train_samples=1200,
            test_samples=300,
            file_path="ml/models/irrigation_model.joblib",
            is_active=True
        )
        db.session.add(mv)

    db.session.commit()
    print("Database seeding completed successfully!")
