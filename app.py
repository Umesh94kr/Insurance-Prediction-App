from data_validation import UserInput
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

# load model 
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

@app.get('/')
def home():
    return JSONResponse(status_code=200, content={'messages' : "Hey this is home page for our API"})

@app.post('/predict')
def predict_premium(data: UserInput):

    input_df = pd.DataFrame([{
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }])

    prediction = model.predict(input_df)[0]

    return JSONResponse(status_code=200, content={'predicted_category': prediction})