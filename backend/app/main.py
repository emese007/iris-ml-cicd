from pathlib import Path
from typing import List
import logging

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,  # or DEBUG if you want more details
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger("iris-backend")

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "model" / "iris_model.pkl"

app = FastAPI()

logger.info(f"Loading model from {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
logger.info("Model loaded successfully")


class IrisFeatures(BaseModel):
    features: List[float]  # [sepal_length, sepal_width, petal_length, petal_width]


@app.get("/")
def read_root():
    logger.info("GET / called (healthcheck)")
    return {"status": "ok"}


@app.post("/predict")
def predict(data: IrisFeatures):
    logger.info(f"POST /predict with data: {data.features}")
    preds = model.predict([data.features])
    prediction = int(preds[0])
    logger.info(f"Prediction result: {prediction}")
    return {"prediction": prediction}
