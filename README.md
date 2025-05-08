# MLOps Pipeline: Fraud Detection with Progressive Delivery

A production-grade MLOps pipeline demonstrating modern ML deployment practices with a focus on progressive delivery and monitoring. This project showcases:

## Core Features
- **Model Training & Tracking**: 
  - XGBoost classifier for fraud detection
  - MLflow for experiment tracking and model registry
  - PostgreSQL backend for metadata
  - MinIO (S3-compatible) for artifact storage

- **Model Serving**: 
  - FastAPI for real-time predictions
  - Health checks and metrics endpoints
  - Model loading from MLflow registry

- **Progressive Delivery**: 
  - Canary deployments with Flagger
  - Automated rollback based on metrics
  - Traffic shifting based on performance

- **Model Monitoring**: 
  - Real-time drift detection with Evidently AI
  - Performance tracking and alerting
  - Model retraining triggers

## Tech Stack
### ML Pipeline
- **Training**: XGBoost, pandas, scikit-learn
- **Tracking**: MLflow with PostgreSQL backend
- **Storage**: MinIO (S3-compatible)

### Infrastructure
- **API**: FastAPI
- **Container**: Docker
- **Orchestration**: Kubernetes, Flagger
- **Monitoring**: Evidently AI, Prometheus
- **IaC**: Helm

## Project Structure
```
mlops-k8s-pipeline/
├── api/                 # Model serving
│   ├── main.py         # FastAPI implementation
│   └── Dockerfile      # API container build
├── k8s/                # Kubernetes manifests
│   ├── api/            # API deployments
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   ├── mlflow/         # MLflow setup
│   │   └── deployment-service.yaml
│   ├── monitoring/     # Evidently configs
│   │   └── evidently/
│   │       ├── deployment.yaml
│   │       └── configmap.yaml
│   └── service-mesh/   # Flagger configs
│       └── flagger/
│           └── canary.yaml
└── .github/            # CI/CD workflows
    └── workflows/      # GitHub Actions
        └── deploy-infra.yaml
```

## Current Status

### Completed ✅
- Model Development Pipeline
  - Training script with XGBoost
  - MLflow experiment tracking
  - Model versioning and registry

- Infrastructure Setup
  - MLflow deployment on K8s
  - PostgreSQL backend
  - MinIO storage integration
  - Basic API deployment

### In Progress 🚧
- Model Monitoring
  - Evidently service setup
  - Drift detection configuration
  - Performance metrics tracking

### Next Steps 📋
- [ ] Complete Evidently service implementation
- [ ] Configure canary deployment thresholds
- [ ] Set up model retraining pipeline
- [ ] Add comprehensive monitoring dashboard

## Setup and Installation

### Prerequisites
- Docker Desktop with Kubernetes enabled
- Helm v3.x
- kubectl configured for local cluster
- Python 3.9+

### Local Development Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlops-k8s-pipeline.git
cd mlops-k8s-pipeline
```

2. Create Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

3. Deploy Infrastructure:
```bash
# Deploy MLflow and dependencies
kubectl apply -f k8s/mlflow/
kubectl apply -f k8s/monitoring/

# Verify deployments
kubectl get pods -n mlops
```

4. Access Services:
- MLflow UI: http://localhost:30500
- FastAPI Swagger: http://localhost:30800/docs
- Evidently Dashboard: http://localhost:30600

## Usage Guide

### Training a New Model
```bash
# Train model and log to MLflow
python model/train.py

# Get the model URI from MLflow UI
export MODEL_URI="models:/fraud_detection/Production"
```

### Deploying Model Updates
1. Update model version in ConfigMap:
```bash
kubectl edit configmap api-config-map -n mlops
```

2. Trigger rolling update:
```bash
kubectl rollout restart deployment fraud-api -n mlops
```

### Monitoring Model Performance
- View model metrics: MLflow UI
- Check drift status: Evidently Dashboard
- Monitor deployment: Kubernetes Dashboard

## API Reference

### Prediction Endpoint
```bash
POST /predict
Content-Type: application/json

{
    "data": [
        [1.0, -0.5, 2.3, ...],  # Features as per training data
    ]
}
```

Response:
```json
{
    "predictions": [0],  # 0: Normal, 1: Fraud
    "latency": 0.0023   # Prediction time in seconds
}
```

### Health Check
```bash
GET /
```

Response:
```json
{
    "status": "ok"
}
```

## Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch:
```bash
git checkout -b feature/your-feature-name
```

3. Make changes and test:
```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/
```

4. Submit Pull Request:
- Describe changes made
- Link relevant issues
- Include test coverage

### Environment Variables
```bash
# MLflow configuration
MLFLOW_TRACKING_URI="http://mlflow:5000"
MLFLOW_S3_ENDPOINT_URL="http://minio:9000"

# MinIO credentials
AWS_ACCESS_KEY_ID="minioadmin"
AWS_SECRET_ACCESS_KEY="minioadmin"

# Model configuration
MODEL_URI="models:/fraud_detection/Production"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
