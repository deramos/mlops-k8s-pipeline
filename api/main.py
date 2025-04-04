import os
from fastapi import FastAPI, Request
import mlflow
import pandas as pd
import time

# Load model from Mlflow
model = mlflow.pyfunc.load_model(os.getenv("MODEL_URI"))

# Start FastAPI app
app = FastAPI(title="Fraud Detection API")


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request):
    payload = await request.json()
    inputs = pd.DataFrame(payload["data"])

    start = time.time()
    predictions = model.predict(inputs)
    latency = time.time() - start

    return {
        "predictions": predictions.tolist(),
        "latency": round(latency, 4)
    }
