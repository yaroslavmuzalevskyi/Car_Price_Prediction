from fastapi import FastAPI
from pydantic import BaseModel
from predict_car_price import main as predict_main


import sklearn
import sys

app = FastAPI()


class InputParams(BaseModel):
    brand: str
    model: str
    year: int
    mileage_in_km: int
    offer_description: str = None
    power_kw: float
    transmission_type: str
    fuel_type: str
    model_id: int


def compute_price(
    brand,
    model,
    year,
    mileage_in_km,
    offer_description,
    power_kw,
    transmission_type,
    fuel_type,
    model_id,
):
    output = predict_main(
        brand,
        model,
        year,
        mileage_in_km,
        offer_description,
        power_kw,
        transmission_type,
        fuel_type,
        model_id,
    )
    return output


@app.post("/get_price")
def get_price(params: InputParams):
    output = compute_price(
        params.brand,
        params.model,
        params.year,
        params.mileage_in_km,
        params.offer_description,
        params.power_kw,
        params.transmission_type,
        params.fuel_type,
        params.model_id,
    )
    return output
