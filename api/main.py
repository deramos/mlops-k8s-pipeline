import os
from fastapi import FastAPI, Request, HTTPException
import mlflow
import pandas as pd
import time

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


@app.post("/api/v1/check_model")
async def check_model(request: Request):
    try:
        webhook_data = await request.json()
        # Make threshold logic clearer
        degradation_threshold = float(webhook_data.get('metadata', {}).get('threshold', 0.1))

        with CHECK_DURATION.time():
            # ...existing code...
            if degradation > degradation_threshold:  # Direct comparison
                MODEL_FAILURES.inc()
                raise HTTPException(
                    status_code=422,
                    detail={
                        "status": "failed",
                        "message": f"Performance degradation {degradation:.2%} exceeds threshold {degradation_threshold:.2%}"
                    }
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
