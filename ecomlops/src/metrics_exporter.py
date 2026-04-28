"""
Custom Prometheus metrics exporter for EcoMLOps.
Reads model metrics and drift status from files and exposes as Prometheus metrics.
"""
import os
import json
import time
from prometheus_client import start_http_server, Gauge, Info

MODEL_ACCURACY = Gauge("ml_model_accuracy", "Current model accuracy")
MODEL_F1 = Gauge("ml_model_f1_score", "Current model F1 score")
TRAINING_TIME = Gauge("ml_training_time_seconds", "Last training time in seconds")
INFERENCE_LATENCY = Gauge("ml_inference_latency_seconds", "Last inference latency")
PREDICTION_CONFIDENCE = Gauge("ml_prediction_confidence", "Average prediction confidence")
DRIFT_DETECTED = Gauge("ml_drift_detected", "1 if drift detected, 0 otherwise")
DRIFT_ALERTS_TOTAL = Gauge("ml_drift_alerts_total", "Total drift alerts")
MODEL_VERSION = Info("ml_model_version", "Current model version")

METRICS_DIR = "/app/metrics"


def update_metrics():
    """Read metrics files and update Prometheus gauges."""
    # Model metrics
    model_file = os.path.join(METRICS_DIR, "model_metrics.json")
    if os.path.exists(model_file):
        with open(model_file) as f:
            data = json.load(f)
            MODEL_ACCURACY.set(data.get("model_accuracy", 0))
            MODEL_F1.set(data.get("model_f1", 0))
            TRAINING_TIME.set(data.get("training_time_seconds", 0))
            version = data.get("model_version", "unknown")
            MODEL_VERSION.info({"version": version})

    # Inference metrics
    inference_file = os.path.join(METRICS_DIR, "inference_metrics.json")
    if os.path.exists(inference_file):
        with open(inference_file) as f:
            data = json.load(f)
            INFERENCE_LATENCY.set(data.get("inference_latency_seconds", 0))
            PREDICTION_CONFIDENCE.set(data.get("prediction_confidence", 0))

    # Drift metrics
    drift_file = os.path.join(METRICS_DIR, "drift_log.json")
    if os.path.exists(drift_file):
        with open(drift_file) as f:
            data = json.load(f)
            DRIFT_DETECTED.set(1 if data.get("drift_detected", False) else 0)
            DRIFT_ALERTS_TOTAL.set(data.get("total_alerts", 0))
    else:
        DRIFT_DETECTED.set(0)
        DRIFT_ALERTS_TOTAL.set(0)


if __name__ == "__main__":
    start_http_server(9100)
    print("Metrics exporter started on port 9100")

    while True:
        update_metrics()
        time.sleep(10)
