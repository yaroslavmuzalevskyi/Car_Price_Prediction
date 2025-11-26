import re

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


df = pd.read_csv("data.csv", index_col=0)

target_column = "price_in_euro"
if target_column not in df.columns:
    raise ValueError(f"Expected target column '{target_column}' not found in the dataset.")

numeric_columns = ["year", target_column, "power_kw", "power_ps", "mileage_in_km"]
for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

def _extract_numeric(value: object) -> float:
    """Extract the first numeric value from a string like '10,5 l/100 km'."""
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

df["fuel_consumption_l_per_100km"] = df["fuel_consumption_l_100km"].apply(_extract_numeric)
df["fuel_consumption_g_per_km"] = df["fuel_consumption_g_km"].apply(_extract_numeric)
df["registration_date_parsed"] = pd.to_datetime(
    df["registration_date"],
    format="%m/%Y",
    errors="coerce"
)
df["registration_year"] = df["registration_date_parsed"].dt.year
df["registration_month"] = df["registration_date_parsed"].dt.month
current_year = pd.Timestamp.now().year
df["car_age"] = (current_year - df["year"]).where(df["year"].notna())
df.loc[df["car_age"] <= 0, "car_age"] = np.nan
df["mileage_per_year"] = df["mileage_in_km"] / df["car_age"]
df["power_kw_per_ps"] = df["power_kw"] / df["power_ps"].replace(0, np.nan)
df = df.drop(columns=["registration_date_parsed"])

df = df.dropna(subset=[target_column])

X = df.drop(columns=[target_column])
y = df[target_column]

numeric_features = [
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
    "fuel_consumption_g_per_km"
]
categorical_features = ["brand", "model", "fuel_type", "color", "transmission_type"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "onehot",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = HistGradientBoostingRegressor(random_state=42)

pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

param_distributions = {
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "model__max_depth": [None, 3, 5, 7],
    "model__max_leaf_nodes": [15, 31, 63, 127],
    "model__min_samples_leaf": [10, 20, 30, 50, 100],
    "model__l2_regularization": [0.0, 0.1, 0.5, 1.0]
}
param_sampler = list(ParameterSampler(param_distributions, n_iter=8, random_state=42))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Starting hyperparameter search over {len(param_sampler)} parameter sets...")
best_params = None
best_cv_mae = np.inf

for idx, params in enumerate(param_sampler, start=1):
    progress = idx / len(param_sampler) * 100
    print(f"[{idx}/{len(param_sampler)} | {progress:.0f}%] Training with params: {params}")
    scores = cross_val_score(
        pipeline.set_params(**params),
        X_train,
        y_train,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=1
    )
    mean_mae = -scores.mean()
    print(f"    -> CV MAE: {mean_mae:.2f}")
    if mean_mae < best_cv_mae:
        best_cv_mae = mean_mae
        best_params = params
        print("    -> New best parameters found!")

print(f"Best params: {best_params}")
print(f"CV MAE: {best_cv_mae:.2f}")

best_pipeline = pipeline.set_params(**best_params)
best_pipeline.fit(X_train, y_train)
joblib.dump(best_pipeline, "car_price_model.joblib")

y_pred = best_pipeline.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
print("Model saved to car_price_model.joblib")
