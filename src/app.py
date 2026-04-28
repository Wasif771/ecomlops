"""
FastAPI application for EcoMLOps model serving.
Provides prediction endpoints with Prometheus metrics instrumentation.
"""
import os
import json
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from predict import predictor

app = FastAPI(
    title="EcoMLOps Iris Classifier",
    description="Lightweight MLOps API with observability",
    version="1.0.0"
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["status"]
)
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)
PREDICTION_CONFIDENCE = Gauge(
    "prediction_confidence",
    "Confidence of latest prediction"
)
MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Current model accuracy"
)


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="Iris features: [sepal_length, sepal_width, petal_length, petal_width]"
    )


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    confidence: float
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction on iris features."""
    start_time = time.time()
    try:
        result = predictor.predict(request.features)
        latency = time.time() - start_time

        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(latency)
        PREDICTION_CONFIDENCE.set(result["confidence"])

        return PredictionResponse(
            prediction=result["prediction"],
            class_name=result["class_name"],
            confidence=result["confidence"],
            latency_ms=result["latency_ms"]
        )
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/batch-predict")
async def batch_predict(sample_size: int = 50):
    """Run batch prediction for drift detection baseline."""
    from sklearn.datasets import load_iris
    iris = load_iris()
    indices = np.random.choice(len(iris.data), size=sample_size, replace=False)
    samples = iris.data[indices]
    result = predictor.predict_batch(samples)
    return result


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Update model accuracy gauge if available
    metrics_file = "/app/metrics/model_metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            data = json.load(f)
            MODEL_ACCURACY.set(data.get("model_accuracy", 0))

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
