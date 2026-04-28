"""
Training script for EcoMLOps pipeline.
Trains a logistic regression model on Iris dataset and logs to MLflow.
"""
import os
import pickle
import json
import time
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("ecomlops-iris-classification")


def train_and_evaluate():
    """Train model and log all metrics/artifacts to MLflow."""
    with mlflow.start_run(run_name="iris-logistic-regression") as run:
        # Load dataset
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )

        # Log parameters
        params = {
            "model_type": "LogisticRegression",
            "max_iter": 200,
            "solver": "lbfgs",
            "random_state": 42,
            "test_size": 0.2,
            "dataset": "iris",
            "n_features": iris.data.shape[1],
            "n_classes": len(np.unique(iris.target))
        }
        mlflow.log_params(params)

        # Train model with timing
        start_time = time.time()
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "training_time_sec": training_time
        }
        mlflow.log_metrics(metrics)

        # Save model artifact
        model_path = "/app/models/iris_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path)

        # Log sklearn model to registry
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="iris-classifier"
        )

        # Save metrics to file for Prometheus to scrape
        metrics_export = {
            "model_accuracy": metrics["accuracy"],
            "model_f1": metrics["f1_macro"],
            "training_time_seconds": training_time,
            "model_version": run.info.run_id[:8]
        }
        os.makedirs("/app/metrics", exist_ok=True)
        with open("/app/metrics/model_metrics.json", "w") as f:
            json.dump(metrics_export, f)

        print(f"Training completed. Accuracy: {metrics['accuracy']:.4f}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Metrics: {metrics}")

        return metrics


if __name__ == "__main__":
    train_and_evaluate()
