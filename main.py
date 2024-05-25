# Put the code for your API here.
import os
import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference


model = pickle.load(open("model/model.pkl", "rb"))
encoder = pickle.load(open("model/onehotencoder.pkl", "rb"))
lb = pickle.load(open("model/labelbinarizer.pkl", "rb"))

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias = "education-num")
    marital_status: str = Field(alias = "marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias = "capital-gain")
    capital_loss: int = Field(alias = "capital-loss")
    hours_per_week: int = Field(alias = "hours-per-week")
    native_country: str = Field(alias = "native-country")
    class Config:
        jason_schema_extra = {
            "example": {
                "age": 59,
                "workclass": "Private",
                "fnlgt": 109015,
                "education": "HS-grad",
                "education-num": 9,
                "marital-status": "Divorced",
                "occupation": "Tech-support",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Female",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
                }
        }


app = FastAPI()

@app.get("/")
def root():
    return {"msg": "Hello World !"}

@app.post("/predict")
def prediction(input: InputData):
    df = pd.DataFrame(input.model_dump(by_alias=True), index = [0])
    df.columns = [x.replace("_", "-") for x in df.columns]
    df = df.assign(salary="Undefined")

    X_test, _, _, _ = process_data(
        df, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=None)

    pred = inference(model, X_test)
    if pred >= 0.5:
        output = ">50K" 
    else:
        output = "<=50K"

    return {"prediction": output}