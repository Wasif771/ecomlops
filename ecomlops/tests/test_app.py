"""
Unit tests for EcoMLOps pipeline.
"""
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import ModelPredictor


def test_model_loading():
    """Test that model loads correctly."""
    predictor = ModelPredictor()
    assert predictor.model is not None


def test_prediction():
    """Test prediction with known Iris sample."""
    predictor = ModelPredictor()
    # Setosa sample
    result = predictor.predict([5.1, 3.5, 1.4, 0.2])
    assert "prediction" in result
    assert "class_name" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1


def test_batch_prediction():
    """Test batch prediction."""
    predictor = ModelPredictor()
    import numpy as np
    samples = np.random.rand(10, 4).tolist()
    result = predictor.predict_batch(samples)
    assert "predictions" in result
    assert len(result["predictions"]) == 10


def test_drift_detector():
    """Test drift detection logic."""
    from drift_detector import DriftDetector

    detector = DriftDetector(threshold=0.7)

    # High confidence - no drift
    assert detector.check_drift(0.95, 0) == False

    # Low confidence - drift detected
    assert detector.check_drift(0.5, 1) == True

    stats = detector.get_drift_stats()
    assert stats["total_checks"] == 2
    assert stats["drift_alerts"] == 1


if __name__ == "__main__":
    test_model_loading()
    test_prediction()
    test_batch_prediction()
    test_drift_detector()
    print("All tests passed!")
