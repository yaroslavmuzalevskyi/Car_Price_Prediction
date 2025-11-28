from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import gradio as gr
import requests

API_BASE_URL = "http://localhost:8000"
ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "y_muz_noAPI" / "data.csv"
MODELS_DIR = ROOT_DIR / "models"

DEFAULT_MODEL_IDS = [2, 6, 12, 48, 256]
KNOWN_TRANSMISSIONS = ["Automatic", "Manual", "Semi-automatic", "Unknown"]
KNOWN_FUEL_TYPES = [
    "Petrol",
    "Diesel",
    "Hybrid",
    "Diesel Hybrid",
    "Electric",
    "LPG",
    "CNG",
    "Other",
    "Unknown",
]

FALLBACK_BRAND_MODELS = {
    "Audi": ["Audi A1", "Audi A3", "Audi A4", "Audi A5", "Audi Q5"],
    "BMW": ["BMW 118", "BMW 318", "BMW 320", "BMW 520", "BMW X5"],
    "Mercedes-Benz": [
        "A 150",
        "A 180",
        "C 200",
        "E 220",
        "GLC 300",
    ],
    "Volkswagen": [
        "Golf",
        "Passat Variant",
        "Tiguan",
        "Polo",
    ],
    "Skoda": ["Fabia", "Octavia", "Superb", "Kodiaq"],
}

DESCRIPTION_PRESETS = [
    ("No extra description", ""),
    ("Well maintained, single owner.", "Well maintained, single owner."),
    (
        "Accident-free with full service history.",
        "Accident-free vehicle with full service history.",
    ),
    (
        "Dealer certified, includes warranty.",
        "Dealer certified vehicle that includes remaining warranty.",
    ),
    ("Needs minor cosmetic work.", "Needs minor cosmetic work but runs well."),
]
DESCRIPTION_PRESET_MAP = dict(DESCRIPTION_PRESETS)
MODEL_FILE_PATTERN = re.compile(r"used_car_price_model_(\d+)\.joblib$")
KNOWN_TRANSMISSION_SET = set(KNOWN_TRANSMISSIONS)
KNOWN_FUEL_SET = set(KNOWN_FUEL_TYPES)
YEAR_CHOICES = [str(year) for year in range(1960, 2024)]
# POWER_CHOICES = [str(power) for power in range(40, 601, 10)]
# MIN_POWER, MAX_POWER = 40, 600
MILEAGE_CHOICES = [str(value) for value in range(0, 400001, 5000)]


@dataclass(frozen=True)
class OptionSets:
    brands: List[str]
    brand_models: Dict[str, List[str]]
    transmissions: List[str]
    fuel_types: List[str]
    top_models: List[str]


def load_option_sets() -> OptionSets:
    brand_models: Dict[str, set] = defaultdict(set)
    transmissions = set()
    fuel_types = set()
    model_counts: Counter[str] = Counter()

    if DATASET_PATH.exists():
        with DATASET_PATH.open() as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                brand = (row.get("brand") or "").strip()
                model = (row.get("model") or "").strip()
                transmission = (row.get("transmission_type") or "").strip()
                fuel = (row.get("fuel_type") or "").strip()

                if brand and model:
                    brand_models[brand].add(model)
                    model_counts[model] += 1

                if transmission in KNOWN_TRANSMISSION_SET:
                    transmissions.add(transmission)

                if fuel in KNOWN_FUEL_SET:
                    fuel_types.add(fuel)

    if not brand_models:
        brand_models = {
            brand: sorted(models) for brand, models in FALLBACK_BRAND_MODELS.items()
        }
    else:
        brand_models = {
            brand: sorted(models) for brand, models in brand_models.items() if models
        }

    brands = sorted(brand_models.keys()) or sorted(FALLBACK_BRAND_MODELS.keys())

    if transmissions:
        transmission_list = sorted(
            transmissions, key=lambda t: KNOWN_TRANSMISSIONS.index(t)
        )
    else:
        transmission_list = KNOWN_TRANSMISSIONS[:]

    if fuel_types:
        fuel_order = {fuel: idx for idx, fuel in enumerate(KNOWN_FUEL_TYPES)}
        fuel_list = sorted(fuel_types, key=lambda f: fuel_order.get(f, len(fuel_order)))
    else:
        fuel_list = KNOWN_FUEL_TYPES[:]

    if model_counts:
        top_models = [model for model, _ in model_counts.most_common(50)]
    else:
        top_models = []
        for models in brand_models.values():
            for model in models:
                if model not in top_models:
                    top_models.append(model)

    return OptionSets(
        brands=brands,
        brand_models=brand_models,
        transmissions=transmission_list,
        fuel_types=fuel_list,
        top_models=top_models,
    )


def get_models_for_brand(brand: str, option_sets: OptionSets) -> List[str]:
    models = option_sets.brand_models.get(brand)
    if models:
        return models
    if option_sets.top_models:
        return option_sets.top_models
    fallback_models = []
    for items in option_sets.brand_models.values():
        fallback_models.extend(items)
    return fallback_models or ["Model not found"]


def discover_model_ids(models_dir: Path = MODELS_DIR) -> List[int]:
    model_ids = []
    if models_dir.exists():
        for path in models_dir.glob("used_car_price_model_*.joblib"):
            match = MODEL_FILE_PATTERN.search(path.name)
            if match:
                model_ids.append(int(match.group(1)))
    model_ids = sorted(set(model_ids))
    return model_ids or DEFAULT_MODEL_IDS


OPTION_SETS = load_option_sets()
AVAILABLE_MODEL_IDS = discover_model_ids()
MODEL_ID_CHOICES = [str(mid) for mid in AVAILABLE_MODEL_IDS]
DESCRIPTION_CHOICES = [label for label, _ in DESCRIPTION_PRESETS]
MODEL_ID_HELP_TEXT = (
    "Model ID equals the suffix of the trained model filename, "
    "e.g. `models/used_car_price_model_50.joblib` → `50`. "
    f"Currently detected IDs: {', '.join(MODEL_ID_CHOICES)}. "
    "Train new models with `python3 y_zab_API/train_used_car_price_model.py "
    "--n-trees <N>` to add more files/IDs."
)
DEFAULT_YEAR_VALUE = "2015" if "2015" in YEAR_CHOICES else YEAR_CHOICES[0]
DEFAULT_POWER_VALUE = "110"
DEFAULT_MILEAGE_VALUE = "120000" if "120000" in MILEAGE_CHOICES else MILEAGE_CHOICES[0]


def update_model_dropdown(selected_brand: str) -> Dict[str, List[str]]:
    """Update the model dropdown based on selected brand."""
    models = get_models_for_brand(selected_brand, OPTION_SETS)
    default_value = models[0] if models else None

    return gr.update(choices=models, value=default_value)


def predict_price(
    brand: str,
    model: str,
    year: Union[int, float],
    power_kw: Union[int, float],
    transmission_type: str,
    fuel_type: str,
    mileage_in_km: Union[int, float],
    model_id: Union[str, int],
    offer_description: str,
) -> Dict[str, Union[str, float, List[float], bool]]:
    """Call the FastAPI backend and return the prediction (or error)."""
    try:
        payload = {
            "brand": (brand or "").strip(),
            "model": (model or "").strip(),
            "year": int(year),
            "power_kw": float(power_kw),
            "transmission_type": transmission_type,
            "fuel_type": fuel_type,
            "mileage_in_km": int(mileage_in_km),
            "model_id": int(model_id),
            "offer_description": DESCRIPTION_PRESET_MAP.get(
                offer_description, offer_description or ""
            ),
        }
    except (TypeError, ValueError) as exc:
        return {"status": False, "error": f"Invalid input: {exc}"}

    required_fields = ["brand", "model", "transmission_type", "fuel_type"]
    missing = [key for key in required_fields if not payload[key]]
    if missing:
        return {
            "status": False,
            "error": f"The following fields must be set: {', '.join(missing)}",
        }

    try:
        response = requests.post(f"{API_BASE_URL}/get_price", json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return {"status": False, "error": f"API request failed: {exc}"}

    if not data.get("status"):
        return {"status": False, "error": data.get("error_message", "Unknown error")}

    return {
        "status": True,
        "predicted_price": round(data["predicted_price"], 2),
        "uncertainty_std": round(data["uncertainty_std"], 2),
        "prediction_interval": [
            round(data["prediction_interval"][0], 2),
            round(data["prediction_interval"][1], 2),
        ],
        "confidence_score": round(data["confidence_score"], 4),
    }


def build_ui() -> gr.Blocks:
    default_brand = OPTION_SETS.brands[0] if OPTION_SETS.brands else ""
    default_models = get_models_for_brand(default_brand, OPTION_SETS)
    default_model_value = default_models[0] if default_models else ""
    default_model_id = MODEL_ID_CHOICES[0] if MODEL_ID_CHOICES else "12"
    default_description = DESCRIPTION_CHOICES[0] if DESCRIPTION_CHOICES else ""

    with gr.Blocks(title="Car Price Prediction") as demo:
        gr.Markdown(
            "### Car Price Prediction UI\n"
            "Train the desired RandomForest models (see `y_zab_API/readme.md`), "
            "run `uvicorn main:app --reload`, and then select the matching model ID "
            "alongside the car details below."
        )

        with gr.Row():
            brand_input = gr.Dropdown(
                label="Brand",
                choices=OPTION_SETS.brands,
                value=default_brand,
                allow_custom_value=False,
            )
            model_input = gr.Dropdown(
                label="Model",
                choices=default_models,
                value=default_model_value,
                allow_custom_value=False,
            )

        with gr.Row():
            year_input = gr.Dropdown(
                label="Year",
                choices=YEAR_CHOICES,
                value=DEFAULT_YEAR_VALUE,
            )
            power_input = gr.Textbox(
                label="Power (kW)",
                value=DEFAULT_POWER_VALUE,
                type="text",
            )

        with gr.Row():
            transmission_input = gr.Dropdown(
                label="Transmission Type",
                choices=OPTION_SETS.transmissions,
                value=OPTION_SETS.transmissions[0],
            )
            fuel_input = gr.Dropdown(
                label="Fuel Type",
                choices=OPTION_SETS.fuel_types,
                value=OPTION_SETS.fuel_types[0],
            )

        mileage_input = gr.Textbox(
            label="Mileage (km)",
            # choices=MILEAGE_CHOICES,
            value=DEFAULT_MILEAGE_VALUE,
        )

        with gr.Row():
            model_id_input = gr.Dropdown(
                label="Model ID",
                choices=MODEL_ID_CHOICES,
                value=default_model_id,
                allow_custom_value=False,
            )
            description_input = gr.Dropdown(
                label="Offer Description",
                choices=DESCRIPTION_CHOICES,
                value=default_description,
                allow_custom_value=False,
            )
        gr.Markdown(MODEL_ID_HELP_TEXT)

        submit_button = gr.Button("Predict price", variant="primary")
        output_json = gr.JSON(label="Prediction")

        brand_input.change(
            fn=update_model_dropdown, inputs=[brand_input], outputs=[model_input]
        )
        submit_button.click(
            fn=predict_price,
            inputs=[
                brand_input,
                model_input,
                year_input,
                power_input,
                transmission_input,
                fuel_input,
                mileage_input,
                model_id_input,
                description_input,
            ],
            outputs=output_json,
        )

    return demo


demo = build_ui()


if __name__ == "__main__":
    demo.launch()
