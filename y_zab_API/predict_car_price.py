import argparse
import joblib
import numpy as np
import pandas as pd

model_set = [2, 6, 12, 48, 256]


def load_model(model_path: str):
    """
    Loading the trained pipeline (preprocessor + RandomForestRegressor).
    """
    m = joblib.load(model_path)
    joblib.dump(m, model_path)
    return joblib.load(model_path)


def predict_with_uncertainty(
    model, X: pd.DataFrame, lower_q: float = 2.5, upper_q: float = 97.5
):
    preprocessor = model.named_steps["preprocessor"]
    rf = model.named_steps["regressor"]

    # Transforming features once
    X_processed = preprocessor.transform(X)

    # Collecting predictions from all trees
    all_tree_preds = np.stack(
        [tree.predict(X_processed) for tree in rf.estimators_],
        axis=0,
    )

    mean_pred = all_tree_preds.mean(axis=0)
    std_pred = all_tree_preds.std(axis=0)

    lower = np.percentile(all_tree_preds, lower_q, axis=0)
    upper = np.percentile(all_tree_preds, upper_q, axis=0)

    interval_width = upper - lower
    confidence = 1.0 - (interval_width / (np.abs(mean_pred) + 1e-8))
    confidence = np.clip(confidence, 0.0, 1.0)

    return mean_pred, std_pred, lower, upper, confidence


def build_input_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    data = {
        "brand": args.brand,
        "model": args.model,
        "year": int(args.year),
        "power_kw": float(args.power_kw),
        "transmission_type": args.transmission_type,
        "fuel_type": args.fuel_type,
        "mileage_in_km": int(args.mileage_in_km),
        "offer_description": args.offer_description or "",
    }

    return pd.DataFrame([data])


def main(
    brand=None,
    model=None,
    year=None,
    mileage_in_km=None,
    offer_description=None,
    power_kw=None,
    transmission_type=None,
    fuel_type=None,
    model_id=None,
):
    external_call = False
    if (
        brand is None
        or model is None
        or year is None
        or mileage_in_km is None
        or power_kw is None
        or transmission_type is None
        or fuel_type is None
        or model_id is None
    ):
        parser = argparse.ArgumentParser(description="Predict used car price")

        parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            default="models/used_car_price_model_50.joblib",
            help="Path to the trained model file.",
        )

        parser.add_argument("--brand", type=str, required=True, help="Car brand.")
        parser.add_argument("--model", type=str, required=True, help="Car model.")
        parser.add_argument(
            "--year", type=int, required=True, help="Year of production."
        )
        parser.add_argument(
            "--power_kw",
            type=float,
            required=True,
            help="Engine power in kW.",
        )
        parser.add_argument(
            "--transmission_type",
            type=str,
            required=True,
            help="Transmission type (e.g. Manual, Automatic).",
        )
        parser.add_argument(
            "--fuel_type",
            type=str,
            required=True,
            help="Fuel type (e.g. Petrol, Diesel, Electric).",
        )
        parser.add_argument(
            "--mileage_in_km",
            type=int,
            required=True,
            help="Mileage in kilometers.",
        )
        parser.add_argument(
            "--offer-description",
            dest="offer_description",
            type=str,
            default="",
            help="Textual description of the offer.",
        )

        args = parser.parse_args()
    else:
        if model_id not in model_set:
            output = {
                "status": False,
                "error_message": f"Invalid model_id {model_id}. Must be one of {str(model_set)}.",
            }
            return output
        args = argparse.Namespace()
        external_call = True
        args.model_path = f"models/used_car_price_model_{model_id}.joblib"
        args.brand = brand
        args.model = model
        args.year = year
        args.power_kw = power_kw
        args.transmission_type = transmission_type
        args.fuel_type = fuel_type
        args.mileage_in_km = mileage_in_km
        args.offer_description = offer_description

    # Load model
    model = load_model(str(args.model_path))

    # Build input
    X = build_input_dataframe(args)

    # Predict
    mean_pred, std_pred, lower, upper, confidence = predict_with_uncertainty(model, X)

    price = mean_pred[0]
    std = std_pred[0]
    low = lower[0]
    high = upper[0]
    conf = confidence[0]

    output = {
        "status": True,
        "predicted_price": price,
        "uncertainty_std": std,
        "prediction_interval": [low, high],
        "confidence_score": conf,
    }

    if external_call == True:
        return output

    print("=== Prediction result ===")
    print(f"Predicted price:          {price:,.2f} EUR")
    print(f"Uncertainty (std dev):    {std:,.2f} EUR")
    print(f"95% prediction interval:  [{low:,.2f}, {high:,.2f}] EUR")
    print(f"Confidence score (0-1):   {conf:.3f}")


if __name__ == "__main__":
    main()
