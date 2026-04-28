"""
Drift detection module for EcoMLOps.
Monitors model predictions and triggers retraining alerts.
"""
import os
import json
import time
import numpy as np
from datetime import datetime

DRIFT_LOG = "/app/metrics/drift_log.json"
ALERT_THRESHOLD = float(os.getenv("DRIFT_CONFIDENCE_THRESHOLD", "0.65"))


class DriftDetector:
    """Simple confidence-based drift detection."""

    def __init__(self, threshold=ALERT_THRESHOLD):
        self.threshold = threshold
        self.history = []
        self.alert_count = 0

    def check_drift(self, confidence, prediction):
        """Check if confidence is below threshold indicating potential drift."""
        is_drift = confidence < self.threshold
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": confidence,
            "prediction": prediction,
            "is_drift": is_drift,
            "threshold": self.threshold
        }
        self.history.append(record)

        if is_drift:
            self.alert_count += 1
            record["alert_number"] = self.alert_count
            self._write_alert(record)

        return is_drift

    def _write_alert(self, record):
        """Write drift alert to file for Prometheus/GitHub Actions monitoring."""
        os.makedirs(os.path.dirname(DRIFT_LOG), exist_ok=True)
        drift_data = {
            "drift_detected": True,
            "latest_alert": record,
            "total_alerts": self.alert_count,
            "timestamp": record["timestamp"]
        }
        with open(DRIFT_LOG, "w") as f:
            json.dump(drift_data, f, indent=2)

    def get_drift_stats(self):
        """Get drift statistics."""
        if not self.history:
            return {"status": "no_data", "total_checks": 0}

        confidences = [h["confidence"] for h in self.history]
        return {
            "status": "drift_detected" if self.alert_count > 0 else "healthy",
            "total_checks": len(self.history),
            "drift_alerts": self.alert_count,
            "mean_confidence": float(np.mean(confidences)),
            "min_confidence": float(np.min(confidences)),
            "threshold": self.threshold
        }


drift_detector = DriftDetector()
