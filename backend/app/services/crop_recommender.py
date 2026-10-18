"""
Agronomic Crop Recommendation Engine
Uses multi-criteria agronomic evaluation considering soil type, pH,
temperature, humidity, water demand, and growing duration.
"""

from app.models.irrigation import CropReference

# Comprehensive agronomic crop database
REFERENCE_CROPS = [
    {
        "name": "Wheat",
        "scientific_name": "Triticum aestivum",
        "category": "Cereal",
        "season": "Rabi (Winter)",
        "opt_temp": (15.0, 25.0),
        "opt_ph": (6.0, 7.5),
        "opt_soils": ["loamy", "clay loam", "sandy loam"],
        "water_demand": "Medium",
        "duration_days": 120,
        "root_depth_cm": 60,
        "default_method": "Sprinkler / Drip",
        "fertilizer_tips": "NPK 120:60:40 kg/ha. Apply 50% N at sowing, remaining split at tillering and flowering.",
        "pest_precautions": "Scout for yellow rust, aphids, and loose smut. Treat seeds with Carbendazim.",
        "risks": "Terminal heat stress during grain filling stage if temperatures spike above 30°C in March."
    },
    {
        "name": "Rice (Paddy)",
        "scientific_name": "Oryza sativa",
        "category": "Cereal",
        "season": "Kharif (Monsoon)",
        "opt_temp": (22.0, 35.0),
        "opt_ph": (5.5, 7.2),
        "opt_soils": ["clay", "clay loam", "silty"],
        "water_demand": "Very High",
        "duration_days": 135,
        "root_depth_cm": 45,
        "default_method": "Basin / Alternate Wetting & Drying (AWD)",
        "fertilizer_tips": "NPK 100:50:50 kg/ha with Zinc Sulfate (25 kg/ha). Top dress urea in 3 splits.",
        "pest_precautions": "Watch for stem borer, brown planthopper (BPH), and blast disease.",
        "risks": "Severe moisture sensitivity during panicle initiation; drought can cause empty grains."
    },
    {
        "name": "Maize (Corn)",
        "scientific_name": "Zea mays",
        "category": "Cereal / Fodder",
        "season": "Kharif / Spring",
        "opt_temp": (18.0, 30.0),
        "opt_ph": (6.0, 7.5),
        "opt_soils": ["loamy", "sandy loam", "clay loam"],
        "water_demand": "Medium",
        "duration_days": 105,
        "root_depth_cm": 70,
        "default_method": "Drip / Furrow",
        "fertilizer_tips": "NPK 120:60:50 kg/ha. Apply nitrogen in three splits (basal, knee-high, tasseling).",
        "pest_precautions": "Monitor for Fall Armyworm (FAW) whorl feeding. Apply neem oil or Spinetoram.",
        "risks": "Susceptible to waterlogging during early seedling stages; good drainage required."
    },
    {
        "name": "Cotton",
        "scientific_name": "Gossypium hirsutum",
        "category": "Cash Crop / Fiber",
        "season": "Kharif",
        "opt_temp": (21.0, 35.0),
        "opt_ph": (6.2, 8.2),
        "opt_soils": ["clay loam", "clay", "loamy"],
        "water_demand": "High",
        "duration_days": 160,
        "root_depth_cm": 90,
        "default_method": "Drip Irrigation",
        "fertilizer_tips": "NPK 100:50:50 kg/ha with boron and magnesium foliar sprays at boll development.",
        "pest_precautions": "Scout for pink bollworm and sucking pests (whitefly, thrips).",
        "risks": "Excessive vegetative growth if over-irrigated or over-fertilized with nitrogen."
    },
    {
        "name": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "category": "Vegetable",
        "season": "All Season / Rabi / Kharif",
        "opt_temp": (18.0, 28.0),
        "opt_ph": (6.0, 7.0),
        "opt_soils": ["loamy", "sandy loam", "clay loam"],
        "water_demand": "Medium",
        "duration_days": 90,
        "root_depth_cm": 50,
        "default_method": "Drip with Fertigation",
        "fertilizer_tips": "NPK 150:100:120 kg/ha via fertigation weekly with calcium nitrate to prevent blossom end rot.",
        "pest_precautions": "Check for early blight, late blight, and fruit borer (Helicoverpa).",
        "risks": "Fruit cracking caused by irregular irrigation intervals."
    },
    {
        "name": "Potato",
        "scientific_name": "Solanum tuberosum",
        "category": "Vegetable / Tuber",
        "season": "Rabi (Winter)",
        "opt_temp": (15.0, 24.0),
        "opt_ph": (5.2, 6.8),
        "opt_soils": ["sandy loam", "loamy"],
        "water_demand": "Medium",
        "duration_days": 90,
        "root_depth_cm": 40,
        "default_method": "Drip / Sprinkler",
        "fertilizer_tips": "NPK 150:100:150 kg/ha. High potash requirement for tuber bulking.",
        "pest_precautions": "Late blight alert during foggy, cool humid weather. Apply Mancozeb prophylactically.",
        "risks": "Tuber rotting under waterlogged condition in heavy clay soils."
    },
    {
        "name": "Onion",
        "scientific_name": "Allium cepa",
        "category": "Vegetable",
        "season": "Kharif / Late Kharif / Rabi",
        "opt_temp": (13.0, 28.0),
        "opt_ph": (6.0, 7.5),
        "opt_soils": ["loamy", "sandy loam", "silty"],
        "water_demand": "Medium",
        "duration_days": 110,
        "root_depth_cm": 35,
        "default_method": "Micro-Sprinkler / Drip",
        "fertilizer_tips": "NPK 100:50:50 kg/ha with sulfur 30 kg/ha to enhance pungency and storage life.",
        "pest_precautions": "Thrips and purple blotch. Spray yellow sticky traps and systemic fungicides.",
        "risks": "Shallow root system makes onion sensitive to both drought and excessive moisture."
    },
    {
        "name": "Soybean",
        "scientific_name": "Glycine max",
        "category": "Oilseed / Legume",
        "season": "Kharif (Monsoon)",
        "opt_temp": (20.0, 32.0),
        "opt_ph": (6.0, 7.5),
        "opt_soils": ["loamy", "clay loam"],
        "water_demand": "Medium",
        "duration_days": 100,
        "root_depth_cm": 60,
        "default_method": "Sprinkler / Furrow",
        "fertilizer_tips": "NPK 20:60:40 kg/ha. Inoculate seeds with Rhizobium culture for atmospheric nitrogen fixation.",
        "pest_precautions": "Girdle beetle and pod borer.",
        "risks": "Susceptible to yellow mosaic virus transmitted by whiteflies."
    },
    {
        "name": "Chickpea (Gram)",
        "scientific_name": "Cicer arietinum",
        "category": "Pulse / Legume",
        "season": "Rabi (Post-monsoon)",
        "opt_temp": (14.0, 26.0),
        "opt_ph": (6.0, 8.0),
        "opt_soils": ["loamy", "clay loam", "sandy loam"],
        "water_demand": "Low",
        "duration_days": 105,
        "root_depth_cm": 50,
        "default_method": "Sprinkler / Light Furrow",
        "fertilizer_tips": "NPK 20:40:20 kg/ha. Low nitrogen requirement due to root nodules.",
        "pest_precautions": "Gram pod borer (Helicoverpa armigera) and Fusarium wilt. Practice crop rotation.",
        "risks": "Excessive irrigation promotes vegetative growth at the expense of pod setting."
    },
    {
        "name": "Mustard",
        "scientific_name": "Brassica juncea",
        "category": "Oilseed",
        "season": "Rabi (Winter)",
        "opt_temp": (12.0, 25.0),
        "opt_ph": (6.0, 7.8),
        "opt_soils": ["loamy", "sandy loam"],
        "water_demand": "Low",
        "duration_days": 115,
        "root_depth_cm": 60,
        "default_method": "Sprinkler / Drip",
        "fertilizer_tips": "NPK 80:40:40 kg/ha with sulfur 20 kg/ha to boost oil percentage.",
        "pest_precautions": "Mustard aphid (Lipaphis erysimi) during cloudy flowering weather.",
        "risks": "Frost damage during flowering and early pod stage."
    },
    {
        "name": "Groundnut (Peanut)",
        "scientific_name": "Arachis hypogaea",
        "category": "Oilseed / Legume",
        "season": "Kharif / Summer",
        "opt_temp": (22.0, 32.0),
        "opt_ph": (5.8, 7.2),
        "opt_soils": ["sandy loam", "loamy"],
        "water_demand": "Medium",
        "duration_days": 110,
        "root_depth_cm": 50,
        "default_method": "Sprinkler / Drip",
        "fertilizer_tips": "NPK 25:50:50 kg/ha plus Gypsum 400 kg/ha at pegging stage for pod development.",
        "pest_precautions": "Tikka disease (leaf spot) and leaf miner.",
        "risks": "Hard clay soil prevents peg penetration into the ground; light friable soil preferred."
    },
    {
        "name": "Sugarcane",
        "scientific_name": "Saccharum officinarum",
        "category": "Cash Crop",
        "season": "Perennial / Annual",
        "opt_temp": (20.0, 36.0),
        "opt_ph": (6.5, 8.0),
        "opt_soils": ["loamy", "clay loam"],
        "water_demand": "Very High",
        "duration_days": 360,
        "root_depth_cm": 100,
        "default_method": "Drip / Furrow",
        "fertilizer_tips": "NPK 250:100:125 kg/ha applied across tillering and grand growth phases.",
        "pest_precautions": "Early shoot borer, red rot, and woolly aphid.",
        "risks": "Extremely high water requirement; requires reliable continuous water source."
    }
]

def recommend_top_crops(soil_type, soil_ph=6.8, current_temp=26.0, current_humidity=50.0, water_availability="Medium"):
    """
    Evaluates reference crops based on multi-criteria agronomic match:
    - Soil compatibility: 35% weight
    - Soil pH match: 20% weight
    - Temperature suitability: 25% weight
    - Water demand vs availability match: 20% weight
    """
    soil_key = soil_type.lower().strip()
    scored_crops = []

    for c in REFERENCE_CROPS:
        score = 0.0

        # 1. Soil match (35 pts)
        if soil_key in c["opt_soils"]:
            score += 35.0
        elif ("loam" in soil_key and any("loam" in s for s in c["opt_soils"])):
            score += 25.0
        else:
            score += 10.0

        # 2. pH match (20 pts)
        ph_min, ph_max = c["opt_ph"]
        if ph_min <= soil_ph <= ph_max:
            score += 20.0
        elif abs(soil_ph - ph_min) <= 0.6 or abs(soil_ph - ph_max) <= 0.6:
            score += 12.0
        else:
            score += 5.0

        # 3. Temperature match (25 pts)
        t_min, t_max = c["opt_temp"]
        if t_min <= current_temp <= t_max:
            score += 25.0
        elif (t_min - 4.0) <= current_temp <= (t_max + 4.0):
            score += 15.0
        else:
            score += 5.0

        # 4. Water availability match (20 pts)
        crop_demand = c["water_demand"]
        if water_availability == "High":
            score += 20.0 if crop_demand in ["Very High", "High", "Medium"] else 12.0
        elif water_availability == "Medium":
            score += 20.0 if crop_demand in ["Medium", "Low"] else (10.0 if crop_demand == "High" else 5.0)
        else: # Low water availability
            score += 20.0 if crop_demand == "Low" else (10.0 if crop_demand == "Medium" else 2.0)

        # Cap score between 0 and 100
        total_score = min(98.5, max(40.0, score))

        if total_score >= 82.0:
            category_label = "Highly Suitable"
        elif total_score >= 68.0:
            category_label = "Moderately Suitable"
        else:
            category_label = "Marginally Suitable"

        scored_crops.append({
            "name": c["name"],
            "scientific_name": c["scientific_name"],
            "category": c["category"],
            "season": c["season"],
            "suitability_score": round(total_score, 1),
            "suitability_category": category_label,
            "preferred_soil_conditions": f"{', '.join(c['opt_soils']).title()} (pH {ph_min} - {ph_max})",
            "ideal_temperature_range": f"{t_min}°C – {t_max}°C",
            "water_requirement": c["water_demand"],
            "growing_duration": f"{c['duration_days']} days",
            "irrigation_method": c["default_method"],
            "fertilizer_guidelines": c["fertilizer_tips"],
            "pest_precautions": c["pest_precautions"],
            "risks_limitations": c["risks"]
        })

    # Sort descending by suitability score
    scored_crops.sort(key=lambda x: x["suitability_score"], reverse=True)
    return scored_crops[:5]
