from flask import Flask, request, jsonify
import requests
import pandas as pd
from datetime import datetime
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# Load the trained model and column names
model = joblib.load("crop_connect.pkl")
X_columns = joblib.load("X_columns.pkl")

def get_climate_zone(city_or_state):
    climate_zones = {
        "Himalayan": ["Jammu & Kashmir", "Ladakh", "Himachal Pradesh", "Uttarakhand", "Arunachal Pradesh", "Sikkim"],
        "Temperate": ["Punjab", "Haryana", "Delhi", "Uttar Pradesh", "Bihar", "West Bengal", "Jharkhand"],
        "Semi-Arid": ["Rajasthan", "Gujarat", "Madhya Pradesh", "Maharashtra", "Karnataka", "Telangana", "Chhattisgarh"],
        "Arid": ["Western Rajasthan", "Kutch (Gujarat)"],
        "Tropical Wet": ["Kerala", "Tamil Nadu", "Goa", "Assam", "Meghalaya", "Tripura", "Nagaland", "Mizoram", "Manipur", "Odisha", "Andhra Pradesh", "West Bengal"]
    }
    union_territories = {
        "Andaman & Nicobar Islands": "Tropical Wet",
        "Lakshadweep": "Tropical Wet",
        "Chandigarh": "Temperate",
        "Dadra & Nagar Haveli and Daman & Diu": "Tropical Wet",
        "Puducherry": "Tropical Wet"
    }
    for zone, states in climate_zones.items():
        if city_or_state in states:
            return zone
    return union_territories.get(city_or_state, "Unknown")

def convert_to_24hr(time_str):
    return datetime.strptime(time_str, "%I:%M %p").strftime("%H:%M")

def get_weather_data(city, city_or_state):
    api_key = "2de141547c3c4ef3a55192620250904"
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={city}&days=1&aqi=no&alerts=no"
    response = requests.get(url)
    data = response.json()
    if "error" in data:
        return None
    sunrise_24 = convert_to_24hr(data["forecast"]["forecastday"][0]["astro"]["sunrise"])
    sunset_24 = convert_to_24hr(data["forecast"]["forecastday"][0]["astro"]["sunset"])
    sun_hours = (datetime.strptime(sunset_24, "%H:%M") - datetime.strptime(sunrise_24, "%H:%M")).seconds / 3600
    today = datetime.today()
    return {
        "Climate Zone": get_climate_zone(city_or_state),
        "Month": today.month,
        "Week": int(today.strftime("%U")),
        "Temperature": data["current"]["temp_c"],
        "Humidity": data["current"]["humidity"],
        "Rainfall": data["forecast"]["forecastday"][0]["day"]["totalprecip_mm"],
        "Wind Speed": data["current"]["wind_kph"],
        "Sunlight": round(sun_hours, 2)
    }

def prepare_farm_data(city, city_or_state, crop, growth_stage, soil_type):
    weather_data = get_weather_data(city, city_or_state)
    if not weather_data:
        return None
    farm_data = pd.DataFrame({
        "Crop": [crop],
        "Growth Stage": [growth_stage],
        "Soil Type": [soil_type],
        **{key: [value] for key, value in weather_data.items()}
    })
    return farm_data
    print(farm_data)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    city = data.get("city")
    state = data.get("state")
    crop = data.get("crop")
    growth_stage = data.get("growth_stage")
    soil_type = data.get("soil_type")

    farm_data = prepare_farm_data(city, state, crop, growth_stage, soil_type)
    if farm_data is None:
        return jsonify({"error": "Failed to fetch weather data."}), 400

    farm_data_encoded = pd.get_dummies(farm_data, columns=["Crop", "Growth Stage", "Soil Type", "Climate Zone"])
    farm_data_encoded = farm_data_encoded.reindex(columns=X_columns, fill_value=0)

    predicted_moisture = model.predict(farm_data_encoded)
    return jsonify({"Predicted Ideal Moisture": f"{predicted_moisture[0]}%"})

if __name__ == "__main__":
    app.run(debug=True)
