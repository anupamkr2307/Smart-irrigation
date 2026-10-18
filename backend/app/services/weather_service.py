"""
Weather and Geocoding Service
Interfaces with Open-Meteo API for real-time and forecast agro-meteorological data,
implements in-memory caching with TTL, and falls back gracefully to manual input.
"""

import time
import requests
from datetime import datetime
from app.services.water_service import calculate_penman_et0

# Simple TTL Cache
_WEATHER_CACHE = {}
CACHE_TTL_SECONDS = 3600  # 1 hour cache

def geocode_location(location_name):
    """
    Geocodes city, village, or district using Open-Meteo Geocoding API.
    """
    clean_query = location_name.strip()
    cache_key = f"geo_{clean_query.lower()}"
    now = time.time()

    if cache_key in _WEATHER_CACHE:
        entry = _WEATHER_CACHE[cache_key]
        if now - entry["timestamp"] < CACHE_TTL_SECONDS:
            return entry["data"]

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": clean_query, "count": 5, "language": "en", "format": "json"}

    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                best = results[0]
                res = {
                    "name": best.get("name"),
                    "latitude": best.get("latitude"),
                    "longitude": best.get("longitude"),
                    "state": best.get("admin1", ""),
                    "country": best.get("country", ""),
                    "source": "Open-Meteo Geocoding"
                }
                _WEATHER_CACHE[cache_key] = {"timestamp": now, "data": res}
                return res
    except Exception as e:
        print(f"Geocoding network exception: {e}")

    # Fallback default coordinates if offline or failed
    fallback = {
        "name": clean_query,
        "latitude": 19.9975,
        "longitude": 73.7898,
        "state": "Maharashtra",
        "country": "India",
        "source": "Local Fallback"
    }
    return fallback

def fetch_weather_data(latitude, longitude, location_name="Current Location"):
    """
    Fetches real-time weather and 7-day forecast from Open-Meteo.
    """
    cache_key = f"weather_{round(latitude, 3)}_{round(longitude, 3)}"
    now = time.time()

    if cache_key in _WEATHER_CACHE:
        entry = _WEATHER_CACHE[cache_key]
        if now - entry["timestamp"] < CACHE_TTL_SECONDS:
            return entry["data"]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"],
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "et0_fao_evapotranspiration", "shortwave_radiation_sum"],
        "timezone": "auto"
    }

    try:
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            curr = data.get("current", {})
            daily = data.get("daily", {})

            temp = float(curr.get("temperature_2m", 27.0))
            humidity = float(curr.get("relative_humidity_2m", 50.0))
            wind = float(curr.get("wind_speed_10m", 2.2))
            recent_rain = float(curr.get("precipitation", 0.0))

            # 3-day forecast rainfall sum
            daily_precip = daily.get("precipitation_sum", [0.0, 0.0, 0.0])
            forecast_rain_3d = sum(daily_precip[:3]) if daily_precip else 0.0

            # Daily ET0
            daily_et0 = daily.get("et0_fao_evapotranspiration", [])
            if daily_et0 and len(daily_et0) > 0 and daily_et0[0] is not None:
                et0 = float(daily_et0[0])
            else:
                et0 = calculate_penman_et0(temp, humidity, wind)

            solar_rad_list = daily.get("shortwave_radiation_sum", [])
            solar_rad = float(solar_rad_list[0]) if solar_rad_list and solar_rad_list[0] else 16.0

            weather_obj = {
                "location_name": location_name,
                "latitude": latitude,
                "longitude": longitude,
                "temperature_c": round(temp, 1),
                "relative_humidity_pct": round(humidity, 1),
                "wind_speed_ms": round(wind, 1),
                "recent_rainfall_mm": round(recent_rain, 1),
                "forecast_rainfall_3d_mm": round(forecast_rain_3d, 1),
                "et0_mm_day": round(et0, 2),
                "solar_radiation_mj": round(solar_rad, 1),
                "daily_forecast": [
                    {
                        "date": daily.get("time", [])[idx] if idx < len(daily.get("time", [])) else f"Day {idx+1}",
                        "max_temp": daily.get("temperature_2m_max", [])[idx] if idx < len(daily.get("temperature_2m_max", [])) else temp + 3,
                        "min_temp": daily.get("temperature_2m_min", [])[idx] if idx < len(daily.get("temperature_2m_min", [])) else temp - 4,
                        "rain_mm": daily.get("precipitation_sum", [])[idx] if idx < len(daily.get("precipitation_sum", [])) else 0.0
                    }
                    for idx in range(min(5, len(daily.get("time", []))))
                ],
                "source": "Open-Meteo Live API",
                "timestamp": datetime.utcnow().isoformat()
            }

            _WEATHER_CACHE[cache_key] = {"timestamp": now, "data": weather_obj}
            return weather_obj
    except Exception as e:
        print(f"Weather API exception: {e}")

    # Fallback realistic weather data
    default_weather = {
        "location_name": location_name,
        "latitude": latitude,
        "longitude": longitude,
        "temperature_c": 28.5,
        "relative_humidity_pct": 52.0,
        "wind_speed_ms": 2.4,
        "recent_rainfall_mm": 0.0,
        "forecast_rainfall_3d_mm": 5.0,
        "et0_mm_day": 4.5,
        "solar_radiation_mj": 17.5,
        "daily_forecast": [
            {"date": "Day 1", "max_temp": 31.0, "min_temp": 21.0, "rain_mm": 2.0},
            {"date": "Day 2", "max_temp": 32.0, "min_temp": 22.0, "rain_mm": 3.0},
            {"date": "Day 3", "max_temp": 30.5, "min_temp": 20.5, "rain_mm": 0.0}
        ],
        "source": "Offline Estimated Baseline",
        "timestamp": datetime.utcnow().isoformat()
    }
    return default_weather
