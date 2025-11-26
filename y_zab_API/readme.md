# Before running API

## Training missing models

`python3 train_used_car_price_model.py --n-trees 50 --max-text-features 5000`

* `--n-trees` is a mandatory parameter, it stands for the number of trees in the RandomForest.

* `--max-text-features` is an optional parameter (defaults to 5000), it stands for max features for TfidfVectorizer.

**After running the script a model named `used_car_price_model_50.joblib` will be saved to /models**

Repeat the process as many times as you need with different `--n-trees` paramenter value.

# Running the APi

1. Open **predict_car_price.py** and write all numbers for models you have to `model_set`. For example: if you have 2 models: **used_car_price_model_12.joblib** and **used_car_price_model_25**, your `model_set = [12, 25]`
2. Run `uvicord main:app --reload` in the terminal
3. Send a POST request with details to 127.0.0.1:8000/get_price formatted like follows:
   
{
    "brand": "Audi",
    "model": "Audi A5",
    "year": 2012,
    "mileage_in_km": 150000,
    "offer_description": "40 TFSI",
    "power_kw": 150,
    "transmission_type": "Automatic",
    "fuel_type": "Petrol",
    "model_id": 12
}

`model_id` parameter is used to select the model for processing, for example if `"model_id": 25`, then the script will use **models/used_car_price_model_25.joblib** and so on.