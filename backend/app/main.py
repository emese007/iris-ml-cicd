from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "iris_model.pkl"

app = FastAPI()

model = joblib.load(MODEL_PATH)


class IrisFeatures(BaseModel):
    features: List[float]  # [sepal_length, sepal_width, petal_length, petal_width]


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: IrisFeatures):
    preds = model.predict([data.features])
    return {"prediction": int(preds[0])}