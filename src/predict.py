"""
Prediction module for EcoMLOps pipeline.
Loads trained model and provides prediction interface.
"""
import os
import pickle
import json
import time
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "models/iris_model.pkl")
METRICS_PATH = "/app/metrics/inference_metrics.json"


class ModelPredictor:
    """Simple model predictor with inference metrics tracking."""

    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the latest trained model."""
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
        else:
            # Train a default model if none exists
            from sklearn.datasets import load_iris
            from sklearn.linear_model import LogisticRegression
            iris = load_iris()
            self.model = LogisticRegression(max_iter=200, random_state=42)
            self.model.fit(iris.data, iris.target)
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(self.model, f)

    def predict(self, features):
        """Make prediction and record latency metrics."""
        start_time = time.time()
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        confidence = float(np.max(self.model.predict_proba(features)))
        latency = time.time() - start_time

        # Export inference metrics for Prometheus
        metrics = {
            "inference_latency_seconds": latency,
            "prediction_confidence": confidence,
            "prediction_class": int(prediction)
        }
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f)

        return {
            "prediction": int(prediction),
            "confidence": confidence,
            "latency_ms": round(latency * 1000, 2),
            "class_name": ["setosa", "versicolor", "virginica"][int(prediction)]
        }

    def predict_batch(self, features_list):
        """Batch prediction for drift detection."""
        predictions = self.model.predict(features_list)
        confidences = np.max(self.model.predict_proba(features_list), axis=1)
        return {
            "predictions": predictions.tolist(),
            "mean_confidence": float(np.mean(confidences)),
            "min_confidence": float(np.min(confidences))
        }


predictor = ModelPredictor()
