import os
import glob
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib
import kagglehub

import argparse

# N_ESTIMATORS = 15  # Number of trees in the RandomForest
N_MAX_TEXT_FEATURES = 5000  # Max features for TfidfVectorizer


# Data loading


def load_raw_dataset() -> pd.DataFrame:
    """
    Download the Kaggle dataset via kagglehub and load the single CSV file.
    """
    path = kagglehub.dataset_download("wspirat/germany-used-cars-dataset-2023")
    print("Path to dataset files:", path)

    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {path}")

    data_path = csv_files[0]
    print("Using data file:", data_path)

    df = pd.read_csv(data_path)
    print(f"Loaded shape: {df.shape}")
    return df


# Column standardization


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:

    rename_map = {
        "Brand": "brand",
        "Model": "model",
        "Year of Production": "year",
        "Price in Euro": "price_in_euro",
        "Power in kW": "power_kw",
        "Power in PS": "power_ps",
        "Transmission Type": "transmission_type",
        "Fuel Type": "fuel_type",
        "Mileage": "mileage_in_km",
        "Offer Description": "offer_description",
    }

    for old_name, new_name in rename_map.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    return df


def ensure_power_kw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is a 'power_kw' column, convert from PS if necessary.
    """
    if "power_kw" in df.columns:
        return df

    if "power_ps" in df.columns:
        df["power_kw"] = df["power_ps"].astype(float) * 0.735499
        return df

    raise KeyError("Neither 'power_kw' nor 'Power in PS'/'power_ps' found in dataset.")


def prepare_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, list, list]:
    df = standardize_column_names(df)
    df = ensure_power_kw(df)

    required_columns = [
        "brand",
        "model",
        "year",
        "price_in_euro",
        "power_kw",
        "transmission_type",
        "fuel_type",
        "mileage_in_km",
        "offer_description",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    df = df[required_columns].copy()

    # Making price numeric, dropping non-numeric rows
    df["price_in_euro"] = pd.to_numeric(df["price_in_euro"], errors="coerce")
    df = df.dropna(subset=["price_in_euro"])

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["power_kw"] = pd.to_numeric(df["power_kw"], errors="coerce")
    df["mileage_in_km"] = pd.to_numeric(df["mileage_in_km"], errors="coerce")

    df["offer_description"] = df["offer_description"].fillna("").astype(str)

    # Separating target and features
    y = df["price_in_euro"]
    X = df.drop(columns=["price_in_euro"])

    numeric_features = ["year", "power_kw", "mileage_in_km"]
    categorical_features = ["brand", "model", "transmission_type", "fuel_type"]

    return X, y, numeric_features, categorical_features


# Pipeline


def build_model(
    numeric_features: list,
    categorical_features: list,
) -> Pipeline:
    """
    Building the full preprocessing and RandomForest model pipeline.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-trees",
        type=int,
        required=True,
        help="Number of trees in the RandomForest.",
    )
    parser.add_argument(
        "--max-text-features",
        type=int,
        required=False,
        help="Max features for TfidfVectorizer.",
        default=N_MAX_TEXT_FEATURES,
    )
    args = parser.parse_args()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    text_transformer = TfidfVectorizer(
        max_features=args.max_text_features,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("text", text_transformer, "offer_description"),
        ]
    )

    regressor = RandomForestRegressor(
        n_estimators=args.n_trees,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1,
        verbose=2,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    return model, args.n_trees


# Uncertainty estimation


def predict_with_uncertainty(
    model: Pipeline,
    X: pd.DataFrame,
    lower_q: float = 2.5,
    upper_q: float = 97.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        mean_pred:  (n_samples,) mean price prediction
        std_pred:   (n_samples,) standard deviation across trees
        lower:      (n_samples,) lower bound (e.g. 2.5th percentile)
        upper:      (n_samples,) upper bound (e.g. 97.5th percentile)
        confidence: (n_samples,) 0-1 heuristic certainty score
                    (the smaller interval => the higher confidence)
    """

    # Splitting the pipeline to get preprocessor and the underlying RandomForest
    preprocessor = model.named_steps["preprocessor"]
    rf: RandomForestRegressor = model.named_steps["regressor"]

    # Transforming features once
    X_processed = preprocessor.transform(X)

    # Collecting predictions of individual trees
    all_tree_preds = np.stack(
        [tree.predict(X_processed) for tree in rf.estimators_], axis=0
    )

    mean_pred = all_tree_preds.mean(axis=0)
    std_pred = all_tree_preds.std(axis=0)

    lower = np.percentile(all_tree_preds, lower_q, axis=0)
    upper = np.percentile(all_tree_preds, upper_q, axis=0)

    interval_width = upper - lower
    confidence = 1.0 - (interval_width / (np.abs(mean_pred) + 1e-8))
    confidence = np.clip(confidence, 0.0, 1.0)

    return mean_pred, std_pred, lower, upper, confidence


def train_and_evaluate() -> Pipeline:
    df_raw = load_raw_dataset()
    X, y, numeric_features, categorical_features = prepare_dataset(df_raw)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    model, n_trees = build_model(numeric_features, categorical_features)

    print("Fitting model...")
    model.fit(X_train, y_train)

    print("Evaluating on validation set...")
    y_pred = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, y_pred)

    mse = mean_squared_error(y_valid, y_pred)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_valid, y_pred)

    print(f"MAE:  {mae:,.2f} EUR")
    print(f"RMSE: {rmse:,.2f} EUR")
    print(f"R²:   {r2:.4f}")

    # Saveing the trained pipeline
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"used_car_price_model_{n_trees}.joblib")
    joblib.dump(model, model_path)
    print(f"Saved trained model to: {model_path}")

    return model


def example_prediction(model: Pipeline):

    example_car = pd.DataFrame(
        [
            {
                "brand": "Volkswagen",
                "model": "Golf",
                "year": 2018,
                "power_kw": 85,
                "transmission_type": "Manual",
                "fuel_type": "Petrol",
                "mileage_in_km": 60000,
                "offer_description": "Well maintained VW Golf, one owner, full service history.",
            }
        ]
    )

    mean_pred, std_pred, lower, upper, confidence = predict_with_uncertainty(
        model, example_car
    )

    print("\nExample prediction:")
    print(f"Predicted price:   {mean_pred[0]:,.2f} EUR")
    print(f"Std deviation:     {std_pred[0]:,.2f} EUR (uncertainty score)")
    print(f"95% interval:      [{lower[0]:,.2f}, {upper[0]:,.2f}] EUR")
    print(f"Heuristic confidence (0-1): {confidence[0]:.3f}")


if __name__ == "__main__":
    trained_model = train_and_evaluate()
    example_prediction(trained_model)
