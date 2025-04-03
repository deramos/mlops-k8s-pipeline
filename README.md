# Production Ready ML Pipeline
Simple production-ready end-to-end ML pipeline with CI/CD, auto-deployments, rollback, and monitoring. This project 
demonstrates a complete MLOps pipeline for fraud detection using the **Credit Card Fraud Detection Dataset**, deployed 
and orchestrated with **Kubernetes**. It includes model training, experiment tracking, containerized deployment, and 
CI/CD, infrastructure setup, and monitoring.


## Stack: 
FastAPI + Docker + Kubernetes + GitHub Actions + Helm + AWS + Terraform

# 🧠 MLOps K8s Fraud Detection Pipeline


---

## ✅ Current Progress

### 🧱 Core Components
- **Model Training**: XGBoost classifier trained on the Credit Card Fraud Detection dataset.
- **Experiment Tracking**: MLflow running locally and on Kubernetes, logging metrics, parameters, and model artifacts.
- **Model Logging**: MLflow autologging and custom metrics (AUC, recall, confusion matrix).
- **Model Saving**: Trained model saved to `model/model.pkl` using `joblib`.
- **MLflow on Kubernetes**:
  - Deployed using official Docker image: `ghcr.io/mlflow/mlflow`
  - Backend: PostgreSQL via Helm (`bitnami/postgresql`)
  - Artifact Store: MinIO (S3-compatible) via Helm
  - Service exposed via **NodePort** (`localhost:30500`)

---

## 🗂️ Project Structure
```
project_root/
├── README.md
├── data/
│   └── creditcard.csv
├── model/
│   ├── train.py
│   └── model.pkl
├── api/
│   └── main.py (TBD)
├── docker/
│   └── Dockerfile (TBD)
├── k8s/
│   ├── deployment.yaml (TBD)
│   ├── service.yaml (TBD)
│   └── ingress.yaml (Optional)
├── helm/
│   └── (Helm charts for MLflow/TBD)
├── terraform/
│   └── (TBD)
├── .github/
│   └── workflows/ci-cd.yml (TBD)
├── monitoring/
│   ├── prometheus.yaml (TBD)
│   └── grafana-dashboard.json (TBD)
└── mlflow/
    ├── deployment-service.yaml
    └── config/
        └── mlflow-config.yaml
```

---

## 🚀 Running MLflow on Kubernetes

### 1. Deploy PostgreSQL:
```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install postgres bitnami/postgresql \
  --set auth.username=mlflow \
  --set auth.password=mlflow \
  --set auth.database=mlflow
```

### 2. Deploy MinIO:
```bash
helm repo add minio https://charts.min.io/
helm install minio minio/minio \
  --set accessKey=minioadmin \
  --set secretKey=minioadmin
```

### 3. Deploy MLflow:
Apply `mlflow/deployment-service.yaml`

### 4. Access MLflow UI:
Visit: `http://localhost:30500` (via NodePort)

---

## 📦 Next Steps
- [ ] Build FastAPI service to serve the model (`api/main.py`)
- [ ] Containerize model serving with Docker
- [ ] Add CI/CD pipeline (GitHub Actions)
- [ ] Deploy FastAPI + model to Kubernetes
- [ ] Add monitoring with Prometheus/Grafana
- [ ] Simulate and detect data drift

---

## 📊 Dataset
- Dataset: [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Target variable: `Class` (1 = Fraud, 0 = Not Fraud)
- Highly imbalanced (~0.17% fraud)

---

## 📜 License
MIT License
