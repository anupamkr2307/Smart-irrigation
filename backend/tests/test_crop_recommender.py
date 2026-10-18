import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.crop_recommender import recommend_top_crops

def test_crop_recommendation_top_5():
    recs = recommend_top_crops(
        soil_type="loamy",
        soil_ph=6.8,
        current_temp=24.0,
        current_humidity=50.0,
        water_availability="Medium"
    )
    assert len(recs) == 5
    # Verify sorted descending by suitability score
    scores = [r["suitability_score"] for r in recs]
    assert scores == sorted(scores, reverse=True)
    # Check fields in recommendation
    first = recs[0]
    assert "name" in first
    assert "preferred_soil_conditions" in first
    assert "ideal_temperature_range" in first
    assert "fertilizer_guidelines" in first
    assert "pest_precautions" in first

def test_crop_recommendation_clay_soils():
    recs = recommend_top_crops(
        soil_type="clay",
        soil_ph=7.2,
        current_temp=28.0,
        water_availability="High"
    )
    # Crops suited to clay (like Rice or Cotton) should rank high
    top_crop_names = [r["name"] for r in recs]
    assert any(c in top_crop_names for c in ["Rice (Paddy)", "Cotton", "Sugarcane"])
