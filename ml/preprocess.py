"""
Dataset Validation and Preprocessing Script
Validates agronomic bounds, removes duplicates, checks missing values,
and prepares features for training.
"""

import os
import pandas as pd
import numpy as np

def validate_and_preprocess(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    initial_count = len(df)
    print(f"Loaded {initial_count} records from {csv_path}")

    # 1. Missing values check
    missing = df.isnull().sum()
    if missing.any():
        print("Missing values detected, imputing...")
        df = df.dropna()

    # 2. Duplicate records check
    duplicates = df.duplicated(subset=["record_id"]).sum()
    if duplicates > 0:
        df = df.drop_duplicates(subset=["record_id"])

    # 3. Agronomic bounds validation
    valid_mask = (
        (df["soil_ph"].between(4.0, 9.5)) &
        (df["soil_moisture_pct"].between(0.0, 100.0)) &
        (df["temperature_c"].between(0.0, 55.0)) &
        (df["relative_humidity_pct"].between(5.0, 100.0)) &
        (df["recent_rainfall_mm"] >= 0.0) &
        (df["forecast_rainfall_mm"] >= 0.0) &
        (df["net_irrigation_req_mm"] >= 0.0) &
        (df["crop_water_requirement_etc_mm"] >= 0.0)
    )
    df = df[valid_mask]
    print(f"Validated records: {len(df)} / {initial_count} passed physical constraints.")

    # 4. Unit consistency: 1 mm depth on 1 m2 = 1 Liter
    diff = np.abs(df["net_irrigation_req_mm"] - df["net_irrigation_req_liters_m2"])
    if (diff > 0.01).any():
        print("Warning: Unit discrepancy detected! Correcting net_irrigation_req_liters_m2 to match mm.")
        df["net_irrigation_req_liters_m2"] = df["net_irrigation_req_mm"]

    return df

if __name__ == "__main__":
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "irrigation_data.csv"))
    cleaned_df = validate_and_preprocess(data_path)
    print(f"Preprocessed dataset ready with {len(cleaned_df)} records.")
