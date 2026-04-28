# EcoMLOps: Lightweight Docker image for ML API
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create directories for metrics and models
RUN mkdir -p /app/metrics /app/models

# Environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/iris_model.pkl
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV PORT=8000

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run training on first start, then serve
CMD ["sh", "-c", "python src/train.py && uvicorn src.app:app --host 0.0.0.0 --port 8000"]
