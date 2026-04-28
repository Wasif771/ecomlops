# EcoMLOps: A Lightweight Cost-Optimized MLOps Pipeline

**A complete MLOps solution on AWS EC2 using Docker Compose — lightweight alternative to Kubernetes for small-scale ML deployment.**

## Overview

EcoMLOps is a research-oriented MLOps pipeline that demonstrates a complete machine learning operations workflow on a single AWS EC2 instance using Docker Compose. It is designed as a **lightweight, cost-effective alternative** to Kubernetes-based MLOps architectures, making production-grade MLOps accessible to small teams, researchers, and educational environments.

### Key Features

- **Lightweight Architecture**: No Kubernetes required — runs entirely on Docker Compose
- **Cost-Optimized**: Designed for t2.micro/t3.small EC2 instances (AWS Free Tier eligible)
- **Observability-Driven**: Integrated Prometheus + Grafana monitoring
- **Automated Retraining**: Drift detection triggers via GitHub Actions webhooks
- **Full CI/CD**: GitHub Actions pipeline for testing, building, and deployment
- **Experiment Tracking**: MLflow for complete experiment reproducibility

## Architecture

```
                    +------------------+
                    |   GitHub Repo    |
                    |  (Source Code)   |
                    +--------+---------+
                             |
                             | Push to main
                             v
                    +--------+---------+
                    |  GitHub Actions  |
                    |   CI/CD Pipeline |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
    +---------+---------+       +-----------+-----------+
    |   Build & Test    |       |   Push to ECR       |
    |   (Docker Image)  |       |   (Container Reg)   |
    +---------+---------+       +-----------+-----------+
              |                             |
              v                             v
    +---------+---------+       +-----------+-----------+
    |   Deploy to EC2   |       |   Pull Latest Image |
    |   (SSH + Compose) |       |                     |
    +---------+---------+       +-----------+-----------+
              |                             |
              +--------------+--------------+
                             |
                             v
              +--------------+--------------+
              |         AWS EC2 Instance      |
              |  +-------------------------+  |
              |  |    Docker Compose Stack  |  |
              |  |                         |  |
              |  |  +------+  +---------+ |  |
              |  |  |ML API|  | MLflow  | |  |
              |  |  |:8000 |  | :5000   | |  |
              |  |  +------+  +---------+ |  |
              |  |                         |  |
              |  |  +-----------+ +------+ |  |
              |  |  |Prometheus | |Grafana| |  |
              |  |  |  :9090    | | :3000| |  |
              |  |  +-----------+ +------+ |  |
              |  |                         |  |
              |  |  +-------------------+  |  |
              |  |  |  Metrics Exporter  |  |  |
              |  |  |      :9100         |  |  |
              |  |  +-------------------+  |  |
              |  +-------------------------+  |
              +-------------------------------+
```

## Quick Start on EC2

### Prerequisites

- AWS Account with Free Tier
- EC2 instance (Ubuntu 22.04 LTS, t2.micro or t3.small)
- Security Group with ports: 22 (SSH), 8000 (API), 3000 (Grafana), 9090 (Prometheus), 5000 (MLflow)

### Step 1: Launch EC2 Instance

```bash
# Instance Configuration:
# AMI: Ubuntu 22.04 LTS
# Type: t2.micro (Free Tier) or t3.small
# Storage: 20 GB gp2
# Security Group Rules:
#   - SSH (22) from My IP
#   - Custom TCP (8000) from Anywhere
#   - Custom TCP (3000) from Anywhere
#   - Custom TCP (9090) from Anywhere
#   - Custom TCP (5000) from Anywhere
```

### Step 2: Run Setup Script

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP

# Download and run setup script
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/ecomlops/main/scripts/setup-ec2.sh | bash
```

### Step 3: Configure AWS Credentials

```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Default region: us-east-1
# Default output: json
```

### Step 4: Upload Project Files

```bash
# From your local machine:
scp -i your-key.pem -r ./ecomlops/* ubuntu@YOUR_EC2_IP:~/ecomlops/
```

### Step 5: Start Services

```bash
cd ~/ecomlops
docker-compose up -d
```

### Step 6: Verify Deployment

```bash
# Check all containers are running
docker-compose ps

# Test API
curl http://localhost:8000/

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### Step 7: Access Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| ML API | http://YOUR_EC2_IP:8000 | - |
| API Docs | http://YOUR_EC2_IP:8000/docs | - |
| MLflow | http://YOUR_EC2_IP:5000 | - |
| Prometheus | http://YOUR_EC2_IP:9090 | - |
| Grafana | http://YOUR_EC2_IP:3000 | admin / ecomlops123 |

## CI/CD Configuration

### GitHub Secrets

Set these secrets in your GitHub repository (Settings > Secrets and variables > Actions):

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key |
| `EC2_HOST` | Your EC2 public IP address |
| `EC2_SSH_KEY` | Contents of your .pem private key file |
| `ECR_REGISTRY` | Your ECR registry URI (e.g., 123456789.dkr.ecr.us-east-1.amazonaws.com) |

### Pipeline Stages

1. **Test**: Runs flake8 linting and pytest with coverage
2. **Build**: Builds Docker image and pushes to Amazon ECR
3. **Deploy**: SSH into EC2, pulls latest image, restarts services
4. **Smoke Test**: Verifies API health and prediction endpoints

## Research Component

### Research Questions

1. **RQ1**: Can a complete MLOps pipeline be effectively deployed on a single EC2 instance without Kubernetes?
2. **RQ2**: What is the cost-performance tradeoff of t2.micro vs t3.small instances for lightweight ML workloads?
3. **RQ3**: How effectively can Prometheus-based drift detection trigger automated retraining via CI/CD webhooks?

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| API Latency (P95) | 95th percentile response time | < 100ms |
| Model Accuracy | Classification accuracy on Iris | > 0.95 |
| Training Time | Time to train and deploy new model | < 30s |
| Infrastructure Cost | Monthly EC2 cost | < $15/month |
| Pipeline Reliability | Successful CI/CD run rate | > 95% |
| Time to Recovery | Time to rollback failed deployment | < 2 min |

### Experimental Results

See the research paper for detailed experimental validation and statistical analysis.

## Project Structure

```
ecomlops/
├── .github/
│   └── workflows/
│       └── mlops-cicd.yml       # CI/CD pipeline
├── monitoring/
│   ├── prometheus.yml             # Prometheus configuration
│   └── grafana/
│       ├── dashboard.yml          # Dashboard provisioning
│       ├── datasource.yml         # Prometheus datasource
│       └── dashboards/
│           └── mlops-dashboard.json  # Grafana dashboard
├── scripts/
│   └── setup-ec2.sh              # EC2 initialization script
├── src/
│   ├── __init__.py
│   ├── app.py                    # FastAPI application
│   ├── config.py                 # Configuration
│   ├── drift_detector.py         # Drift detection logic
│   ├── metrics_exporter.py       # Prometheus metrics exporter
│   ├── predict.py                # Model inference
│   └── train.py                  # Model training with MLflow
├── tests/
│   └── test_app.py               # Unit tests
├── docker-compose.yml            # Full stack orchestration
├── Dockerfile                    # ML API container
├── Dockerfile.exporter           # Metrics exporter container
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Cost Analysis

| Instance | vCPU | RAM | On-Demand/hr | Monthly (730h) |
|----------|------|-----|--------------|----------------|
| t2.micro | 1 | 1 GB | $0.0116 | $8.47 |
| t3.micro | 2 | 1 GB | $0.0104 | $7.59 |
| t3.small | 2 | 2 GB | $0.0208 | $15.18 |

**AWS Free Tier**: 750 hours of t2.micro or t3.micro per month for 12 months.

## Monitoring & Observability

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `ml_model_accuracy` | Gauge | Current model accuracy |
| `ml_model_f1_score` | Gauge | Current F1 score |
| `ml_inference_latency_seconds` | Gauge | Inference latency |
| `ml_prediction_confidence` | Gauge | Average prediction confidence |
| `ml_drift_detected` | Gauge | Drift detection flag |
| `ml_training_time_seconds` | Gauge | Last training duration |
| `prediction_requests_total` | Counter | Total prediction requests |
| `prediction_latency_seconds` | Histogram | Prediction latency distribution |

### Grafana Dashboards

The pre-configured dashboard includes:
- Model performance metrics (accuracy, F1)
- Real-time inference latency
- Prediction confidence tracking
- Drift detection alerts
- Training time monitoring

## License

MIT License - For academic and educational use.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{ecomlops2024,
  title={EcoMLOps: A Lightweight Cost-Optimized MLOps Pipeline with Observability-Driven Automated Retraining on AWS EC2},
  author={Wasif, Muhammad},
  year={2024}
}
```
