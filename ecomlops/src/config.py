"""
Configuration for EcoMLOps pipeline.
"""
import os


class Config:
    """Application configuration with defaults."""

    # API
    PORT = int(os.getenv("PORT", "8000"))
    HOST = os.getenv("HOST", "0.0.0.0")

    # MLflow
    MLFLOW_TRACKING_URI = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )

    # Model
    MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/iris_model.pkl")

    # Drift Detection
    DRIFT_CONFIDENCE_THRESHOLD = float(
        os.getenv("DRIFT_CONFIDENCE_THRESHOLD", "0.65")
    )

    # AWS
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    ECR_REPOSITORY = os.getenv("ECR_REPOSITORY", "ecomlops")

    # Retraining
    RETRAIN_WEBHOOK_URL = os.getenv("RETRAIN_WEBHOOK_URL", "")

    # Monitoring
    PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))
    GRAFANA_PORT = int(os.getenv("GRAFANA_PORT", "3000"))
