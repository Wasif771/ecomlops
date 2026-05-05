# EcoMLOps: A Lightweight Cost-Optimized MLOps Pipeline

> Submitted for the MLOps course, in our MS-AI at National University of Computer and Emerging Sciences.

**A complete MLOps workflow on AWS EC2 using Docker Compose, MLflow, Prometheus, Grafana, GitHub Actions, and Amazon ECR — designed as a lightweight alternative to Kubernetes for small-scale ML deployment.**

---

## Overview

EcoMLOps is a research-oriented MLOps pipeline that demonstrates a complete machine learning lifecycle on a single AWS EC2 instance. The project focuses on a practical problem: many MLOps platforms depend on Kubernetes or managed cloud services, which can be complex and costly for students, researchers, and small teams. EcoMLOps shows that a production-style MLOps workflow can be implemented with fewer moving parts by using Docker Compose on EC2.

The final implementation validates:

- Dockerized ML API deployment with FastAPI
- Experiment tracking and model registry with MLflow
- Monitoring with Prometheus
- Visualization with Grafana
- CI/CD automation with GitHub Actions
- Container image storage with Amazon ECR
- Cloud deployment on AWS EC2
- Cost-performance tradeoff analysis for small vs. larger EC2 instances

---

## Key Features

- **Lightweight architecture**: Runs on a single AWS EC2 instance without Kubernetes.
- **Docker Compose deployment**: Uses Docker Compose to orchestrate the API, metrics exporter, MLflow, Prometheus, and Grafana services.
- **Full CI/CD pipeline**: GitHub Actions automatically runs tests, builds images, pushes to Amazon ECR, deploys to EC2, and performs smoke testing.
- **Experiment tracking**: MLflow tracks model parameters, metrics, artifacts, and registered model versions.
- **Monitoring and observability**: Prometheus scrapes model/application metrics and Grafana visualizes model performance.
- **Cost-aware deployment**: Uses a larger EC2 instance temporarily for build/debug reliability and recommends smaller instances for runtime operation.
- **Academic/research validation**: Includes research questions, experimental results, architecture diagram, cost-performance analysis, and evidence screenshots.

---

## Research Problem

Modern MLOps pipelines often depend on Kubernetes-based orchestration platforms such as Kubeflow or managed services such as SageMaker. These tools are powerful but can be too complex for lightweight ML workloads and educational environments. This project investigates whether a complete MLOps pipeline can be deployed on a single EC2 instance while still supporting core MLOps requirements: experiment tracking, CI/CD, monitoring, visualization, and deployment.

### Research Questions

1. **RQ1:** Can a complete MLOps pipeline, including experiment tracking, monitoring, CI/CD, and deployment, be effectively deployed on a single EC2 instance without Kubernetes orchestration?
2. **RQ2:** What cost-performance tradeoffs appear when running and building the MLOps stack on small versus larger EC2 instances?
3. **RQ3:** Can Prometheus and Grafana provide enough observability to demonstrate model and infrastructure health in a lightweight MLOps setup?

---

## Architecture

The final architecture uses GitHub Actions for CI/CD, Amazon ECR for container image storage, and AWS EC2 with Docker Compose for deployment.

```text
Developer / Student
        |
        v
GitHub Repository: ecomlops
        |
        v
GitHub Actions CI/CD
  ├── Unit Tests
  ├── Build Docker Image
  ├── Push Image to Amazon ECR
  ├── Deploy to EC2 over SSH
  └── Post-Deployment Smoke Test
        |
        v
Amazon ECR: ecomlops image registry
        |
        v
AWS EC2 Instance
  └── Docker Compose Stack
      ├── ml-api            FastAPI service, port 8000
      ├── metrics-exporter  Prometheus exporter, port 9100
      ├── MLflow            Experiment tracking, port 5000
      ├── Prometheus        Metrics scraping, port 9090
      └── Grafana           Dashboard visualization, port 3000
```

### Architecture Diagram

The project includes an architecture diagram showing the complete flow:

```text
GitHub Repository → GitHub Actions CI/CD → Amazon ECR → AWS EC2 → Docker Compose services
```

Recommended file location:

```text
docs/EcoMLOps_Architecture_diagram.png
```

---

## Technology Stack

| Layer | Tool |
|---|---|
| ML model | scikit-learn Logistic Regression |
| Dataset | Iris dataset |
| API serving | FastAPI |
| Experiment tracking | MLflow 2.11.0 |
| Model registry | MLflow Model Registry |
| Monitoring | Prometheus |
| Visualization | Grafana |
| Containerization | Docker |
| Orchestration | Docker Compose |
| CI/CD | GitHub Actions |
| Image registry | Amazon ECR |
| Cloud deployment | AWS EC2 |
| Runtime OS | Ubuntu on EC2 |

---

## Deployed Services

| Service | Container / Role | Port | URL |
|---|---|---:|---|
| ML API | FastAPI prediction service | 8000 | `http://YOUR_EC2_PUBLIC_IP:8000` |
| API Docs | FastAPI Swagger UI | 8000 | `http://YOUR_EC2_PUBLIC_IP:8000/docs` |
| MLflow | Experiment tracking and model registry | 5000 | `http://YOUR_EC2_PUBLIC_IP:5000` |
| Prometheus | Metrics scraping and querying | 9090 | `http://YOUR_EC2_PUBLIC_IP:9090` |
| Grafana | Monitoring dashboard | 3000 | `http://YOUR_EC2_PUBLIC_IP:3000` |
| Metrics Exporter | Custom Prometheus metrics exporter | 9100 | Internal/Prometheus target |

Grafana default login:

```text
Username: admin
Password: ecomlops123
```

---

## Project Structure

```text
ecomlops/
├── .github/
│   └── workflows/
│       └── mlops-cicd.yml          # GitHub Actions CI/CD pipeline
├── docs/                           # Reports, screenshots, architecture diagram
├── monitoring/
│   ├── prometheus.yml              # Prometheus scrape configuration
│   └── grafana/
│       ├── dashboard.yml           # Grafana dashboard provisioning
│       ├── datasource.yml          # Prometheus data source
│       └── dashboards/
│           └── mlops-dashboard.json
├── scripts/
│   └── setup-ec2.sh                # EC2 setup script
├── src/
│   ├── __init__.py
│   ├── app.py                      # FastAPI application
│   ├── config.py                   # Configuration
│   ├── drift_detector.py           # Drift detection logic
│   ├── metrics_exporter.py         # Prometheus custom metrics exporter
│   ├── predict.py                  # Model loading and inference
│   └── train.py                    # Training with MLflow logging
├── tests/
│   └── test_app.py                 # Unit tests
├── docker-compose.yml              # Docker Compose stack
├── Dockerfile                      # ML API image
├── Dockerfile.exporter             # Metrics exporter image
├── requirements.txt                # Python dependencies
└── README.md
```

---

## AWS Setup Summary

### Region and ECR

This implementation was validated in:

```text
AWS Region: eu-north-1
AWS Account ID: 047557961445
ECR Registry: 047557961445.dkr.ecr.eu-north-1.amazonaws.com
ECR Repository: ecomlops
```

Create the ECR repository once:

```bash
aws ecr create-repository \
  --repository-name ecomlops \
  --region eu-north-1
```

If creating from EC2, the EC2 IAM role needs permission such as `AmazonEC2ContainerRegistryFullAccess`. For normal EC2 runtime pull access, `AmazonEC2ContainerRegistryReadOnly` is enough.

### EC2 IAM Role

Recommended EC2 role:

```text
EC2-ECR-ReadOnly-Role
```

Attach:

```text
AmazonEC2ContainerRegistryReadOnly
```

This allows EC2 to pull images from Amazon ECR without storing long-term AWS access keys on the server.

---

## Quick Start on EC2

### 1. SSH into EC2

```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

### 2. Clone the repository

```bash
cd ~
git clone https://github.com/Wasif771/ecomlops.git
cd ~/ecomlops
```

### 3. Build and start the Docker Compose stack

Use Docker Compose v2 syntax:

```bash
docker compose build --no-cache
docker compose up -d
docker compose ps
```

### 4. Verify API

```bash
curl http://localhost:8000/
```

Prediction test:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### 5. Run training and log to MLflow

```bash
docker compose exec ml-api python src/train.py
```

Expected result:

```text
Training completed. Accuracy: 1.0000
Metrics: {'accuracy': 1.0, 'precision_macro': 1.0, 'recall_macro': 1.0, 'f1_macro': 1.0, 'training_time_sec': 0.0067...}
Registered model: iris-classifier v109
```

---

## CI/CD Pipeline

The repository contains a GitHub Actions workflow that validates and deploys the project.

### Pipeline Stages

| Stage | Description | Latest Observed Result |
|---|---|---|
| Unit Tests | Runs pytest and coverage | Passed in 46 s |
| Build & Push to ECR | Builds Docker image and pushes to Amazon ECR | Passed in about 1 min |
| Deploy to EC2 | SSH deployment to EC2 and service restart | Passed in 25 s |
| Post-Deployment Smoke Test | Verifies health and prediction endpoint | Passed in 34 s |

### GitHub Secrets

Set these in:

```text
GitHub Repository → Settings → Secrets and variables → Actions
```

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM access key for GitHub Actions |
| `AWS_SECRET_ACCESS_KEY` | IAM secret key for GitHub Actions |
| `EC2_HOST` | Current EC2 public IP address |
| `EC2_SSH_KEY` | Full private key content from `.pem` file |
| `ECR_REGISTRY` | `047557961445.dkr.ecr.eu-north-1.amazonaws.com` |

### CI/CD Workflow Behavior

After the first manual setup, normal deployment does not require manual `git pull` or manual Docker restart. The flow is:

```text
git push origin main
        ↓
GitHub Actions runs tests
        ↓
Docker image builds and pushes to ECR
        ↓
GitHub Actions connects to EC2
        ↓
Docker Compose services restart
        ↓
Smoke test validates deployment
```

---

## Experimental Results

### Model Training and Tracking Results

The final validated model was trained using Logistic Regression on the Iris dataset and logged in MLflow.

| Category | Metric | Result |
|---|---|---:|
| Model Accuracy | Accuracy | 1.0000 |
| Model Accuracy | Precision Macro | 1.0000 |
| Model Accuracy | Recall Macro | 1.0000 |
| Model Accuracy | F1 Macro | 1.0000 |
| Training Performance | Training Time | 0.0067 sec |
| Experiment Tracking | MLflow Run | Finished |
| Model Registry | Registered Model | `iris-classifier v109` |

Model parameters recorded in MLflow:

| Parameter | Value |
|---|---|
| model_type | LogisticRegression |
| max_iter | 200 |
| solver | lbfgs |
| random_state | 42 |
| test_size | 0.2 |
| dataset | iris |
| n_features | 4 |

### CI/CD Validation Results

| Pipeline Stage | Observed Result | Evidence |
|---|---|---|
| Unit Tests | Passed in 46 s | GitHub Actions check passed |
| Build and Push to ECR | Passed in about 1 min | Docker image built and pushed to ECR |
| Deploy to EC2 | Passed in 25 s | EC2 Docker Compose services restarted |
| Post-Deployment Smoke Test | Passed in 34 s | API health and prediction endpoint validated |

### Monitoring and Observability Results

| Component | Metric or Status | Observed Result |
|---|---|---|
| Prometheus | metrics-exporter target | UP |
| Prometheus | ml-api target | UP |
| Prometheus | prometheus target | UP |
| Prometheus | `ml_model_accuracy` | 1.0 |
| Prometheus | `ml_model_f1_score` | 1.0 |
| Grafana | Dashboard | Created with Prometheus data source |
| MLflow | Experiment dashboard | Runs and model versions visible |

Note: MLflow does not expose a Prometheus `/metrics` endpoint by default. If MLflow is configured as a Prometheus scrape target, it may show `404 NOT FOUND`. This is expected. In this project, MLflow is used for experiment tracking, while Prometheus monitors the ML API, custom metrics exporter, and Prometheus itself.

---

## Monitoring & Observability

### Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `ml_model_accuracy` | Gauge | Current model accuracy |
| `ml_model_f1_score` | Gauge | Current F1 score |
| `ml_inference_latency_seconds` | Gauge | Inference latency |
| `ml_prediction_confidence` | Gauge | Average prediction confidence |
| `ml_drift_detected` | Gauge | Drift detection flag |
| `ml_training_time_seconds` | Gauge | Last training duration |
| `prediction_requests_total` | Counter | Total prediction requests |
| `prediction_latency_seconds` | Histogram | Prediction latency distribution |

### Prometheus Queries

Open:

```text
http://YOUR_EC2_PUBLIC_IP:9090/graph
```

Try:

```text
ml_model_accuracy
ml_model_f1_score
ml_prediction_confidence
ml_training_time_seconds
prediction_requests_total
```

### Grafana Setup

Open:

```text
http://YOUR_EC2_PUBLIC_IP:3000
```

Login:

```text
admin / ecomlops123
```

Prometheus data source URL:

```text
http://prometheus:9090
```

Recommended dashboard panels:

| Panel | Query | Visualization |
|---|---|---|
| Model Accuracy | `ml_model_accuracy` | Stat or Time series |
| Model F1 Score | `ml_model_f1_score` | Stat or Time series |
| Prediction Confidence | `ml_prediction_confidence` | Gauge |
| Drift Detection | `ml_drift_detected` | Stat |
| Training Time | `ml_training_time_seconds` | Stat |
| Prediction Requests | `prediction_requests_total` | Time series |

Generate prediction traffic before taking dashboard screenshots:

```bash
for i in {1..20}; do
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
  echo
done
```

---

## Cost-Performance Analysis

The cost-performance analysis showed that the lowest-cost EC2 instance is not always reliable for Docker image building. During implementation, a small instance reached approximately 100% CPU utilization and SSH became unresponsive during Docker build. The EBS root volume was increased to 50 GiB gp3 to provide enough space for Docker image layers, build cache, Python dependencies, logs, and ML artifacts. A larger instance, `m7i-flex.large`, was then used temporarily for stable build and debugging.

| Configuration | Purpose | Cost Impact | Performance Observation | Conclusion |
|---|---|---|---|---|
| `t3.micro` + small root disk | Initial low-cost setup | Lowest cost | CPU reached about 100%; SSH became unresponsive during Docker build | Too small for reliable build/deployment |
| 50 GiB gp3 root volume | Docker storage | Slightly higher storage cost | Build cache and Docker layers no longer filled disk | Required for stable Docker workflow |
| `m7i-flex.large` | Build/debug/deploy | Higher temporary cost | Docker build completed successfully; CI/CD deployment passed | Best temporary setup for reliable build |
| Smaller instance after build, e.g. `t3.medium` | Runtime operation | Lower ongoing cost | Expected to be enough for API, MLflow, Prometheus, and Grafana after image is built | Recommended for cost optimization |

Recommended strategy:

```text
Use a larger instance temporarily for build/debug.
Use GitHub Actions and ECR for automated image creation.
Downgrade to a smaller instance such as t3.medium or t3.small for runtime operation.
Stop the EC2 instance when not testing or presenting.
```

---

## Common Commands

### Check running containers

```bash
docker compose ps
```

### View logs

All services:

```bash
docker compose logs -f --tail=100
```

API only:

```bash
docker compose logs -f --tail=100 ml-api
```

Grafana only:

```bash
docker compose logs -f --tail=100 grafana
```

Metrics exporter only:

```bash
docker compose logs -f --tail=100 metrics-exporter
```

### Restart services

```bash
docker compose down
docker compose up -d
```

### Rebuild services

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### ECR login from EC2

```bash
aws ecr get-login-password --region eu-north-1 | \
docker login --username AWS --password-stdin 047557961445.dkr.ecr.eu-north-1.amazonaws.com
```

---

## Implementation Evidence

The final paper and project validation include these evidence items:

1. Architecture diagram for EcoMLOps on AWS EC2.
2. GitHub Actions screenshot showing all checks passed.
3. MLflow experiment run detail and registered model `iris-classifier v109`.
4. Prometheus targets showing `metrics-exporter`, `ml-api`, and `prometheus` as UP.
5. Grafana dashboard showing model accuracy and F1 score.
6. Terminal output showing model training through Docker Compose.
7. Experimental results table.
8. Cost-performance analysis table.

Recommended evidence folder:

```text
docs/evidence/
```

---

## Known Notes and Limitations

- MLflow does not expose `/metrics` for Prometheus by default. Use MLflow for experiment tracking and Prometheus for application/exporter metrics.
- Single EC2 deployment is lightweight and cost-effective, but it creates a single point of failure.
- The Iris dataset and Logistic Regression model are intentionally lightweight for educational reproducibility.
- For larger workloads, Kubernetes, managed services, load balancing, or multi-instance deployment may be required.
- The Git warning inside the training container is not blocking training; it only means Git metadata may not be recorded unless Git is installed in the container.

---

## Research Paper

The final paper is titled:

```text
EcoMLOps: A Lightweight Cost-Optimized MLOps Pipeline with Observability-Driven Monitoring and CI/CD Deployment on AWS EC2
```

It includes:

- Literature review
- Research problem and research questions
- Proposed architecture
- Experimental results
- CI/CD validation
- MLflow tracking results
- Prometheus/Grafana observability results
- Cost-performance analysis
- Implementation evidence appendix
 

---

## License

MIT License - for academic and educational use.
