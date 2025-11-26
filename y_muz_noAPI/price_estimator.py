import re
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib


TARGET_COLUMN = "price_in_euro"

# Columns used in your pipeline
NUMERIC_FEATURES = [
    "year",
    "mileage_in_km",
    "power_kw",
    "power_ps",
    "registration_year",
    "registration_month",
    "car_age",
    "mileage_per_year",
    "power_kw_per_ps",
    "fuel_consumption_l_per_100km",
    "fuel_consumption_g_per_km",
]

CATEGORICAL_FEATURES = ["brand", "model", "fuel_type", "color", "transmission_type"]


def _extract_numeric(value: object) -> float:
    """Same as in training: extract first number from strings like '5,6 l/100 km'."""
    if pd.isna(value):
        return np.nan
    cleaned = str(value).replace(",", ".")
    match = re.search(r"-?\d+(\.\d+)?", cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return np.nan
    return np.nan


def _feature_engineering(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as in the training script."""
    df = raw.copy()

    # numeric
    for col in ["year", "power_kw", "power_ps", "mileage_in_km"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # fuel consumption
    df["fuel_consumption_l_per_100km"] = df["fuel_consumption_l_100km"].apply(_extract_numeric)
    df["fuel_consumption_g_per_km"] = df["fuel_consumption_g_km"].apply(_extract_numeric)

    # registration date
    df["registration_date_parsed"] = pd.to_datetime(
        df["registration_date"], format="%m/%Y", errors="coerce"
    )
    df["registration_year"] = df["registration_date_parsed"].dt.year
    df["registration_month"] = df["registration_date_parsed"].dt.month

    # age & mileage per year
    current_year = pd.Timestamp.now().year
    df["car_age"] = (current_year - df["year"]).where(df["year"].notna())
    df.loc[df["car_age"] <= 0, "car_age"] = np.nan
    df["mileage_per_year"] = df["mileage_in_km"] / df["car_age"]

    # power ratio
    df["power_kw_per_ps"] = df["power_kw"] / df["power_ps"].replace(0, np.nan)

    df = df.drop(columns=["registration_date_parsed"])

    return df


# ---- Load the model once ----
MODEL = joblib.load("car_price_model.joblib")


def estimate_price(car_data: Dict[str, Any]) -> float:
    """
    Take a dict with raw car info and return predicted price in euro.
    car_data must contain at least the columns used below.
    """
    # make a 1-row DataFrame
    raw_df = pd.DataFrame([car_data])

    # same feature engineering as in training
    df = _feature_engineering(raw_df)

    # ensure we only pass the columns the pipeline expects
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    # predict
    price = MODEL.predict(X)[0]
    return float(price)


if __name__ == "__main__":
    example_car = {
        "brand": "Audi",
        "model": "A5 ",
        "fuel_type": "Petrol",
        "color": "black",
        "transmission_type": "Automatic",
        "year": 2012,
        "mileage_in_km": 150000,
        "power_kw": 125,
        "power_ps": 170,
        "fuel_consumption_l_100km": "5,6 l/100 km",
        "fuel_consumption_g_km": "",
        "registration_date": "06/2012",  # mm/YYYY
    }

    price = estimate_price(example_car)
    print(f"Estimated price: {price:,.0f} €")
