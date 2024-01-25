import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from typing import Dict

model_filename = 'coin_linear_regression_model.pkl'
scaler_name = "scaler.pkl"
loaded_model = joblib.load(model_filename)
scaler = joblib.load(scaler_name)
app = FastAPI()

def coin_predict(data_dict):
    
    for key in data_dict:
        data_dict[key] = float(data_dict[key])
    test = np.array([[data_dict["closing_price"], data_dict["opening_price"],data_dict["high_price"],data_dict["low_price"],data_dict["trading_volume"], 100]])
    scaled_test_data = scaler.transform(test)
    predicted_volatality = loaded_model.predict(scaled_test_data[:,:-1])
    original_min = scaler.data_min_
    original_max = scaler.data_max_
    unscaled_data = predicted_volatality[0][0] * (original_max[-1] - original_min[-1]) + original_min[-1]
    return unscaled_data


@app.post("/")
async def process_data(data: Dict):
    result = coin_predict(data)
    return result