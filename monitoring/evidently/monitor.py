from fastapi import FastAPI, HTTPException, Request
from evidently.metrics import DataDriftMetric
from evidently.report import Report
import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
from typing import Dict, Any
from prometheus_client import Counter, Histogram, generate_latest

# Prometheus metrics
MODEL_CHECK_COUNTER = Counter('model_checks_total', 'Number of model checks performed')
MODEL_FAILURES = Counter('model_check_failures', 'Number of failed model checks')
CHECK_DURATION = Histogram('model_check_duration_seconds', 'Time spent performing model check')

app = FastAPI(title="Model Monitoring Service")

class ModelMonitor:
    def __init__(self):
        self.mlflow_client = mlflow.tracking.MlflowClient()
        # Update to use both env variables
        self.drift_threshold = float(os.getenv("DRIFT_THRESHOLD", "0.1"))
        self.performance_threshold = float(os.getenv("PERFORMANCE_THRESHOLD", "0.95"))
        self.reference_data = self._load_reference_data()
        
    def _load_reference_data(self) -> pd.DataFrame:
        """Load reference data directly using mlflow.pyfunc"""
        try:
            # Load production model directly
            production_model = mlflow.pyfunc.load_model("models:/fraud_detection/Production")
            # Get reference data from model's run artifacts
            run_id = production_model.metadata.run_id
            reference_data_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="reference_data.csv"
            )
            return pd.read_csv(reference_data_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load reference data: {str(e)}"
            )

    def get_reference_metrics(self) -> Dict[str, Any]:
        """Get reference metrics directly from model metadata"""
        try:
            # Load production model directly
            production_model = mlflow.pyfunc.load_model("models:/fraud_detection/Production")
            # Get metrics from model metadata
            return production_model.metadata.metrics
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get reference metrics: {str(e)}"
            )

    async def evaluate_current_model(self) -> Dict[str, float]:
        """Evaluate the staging model (canary version)"""
        try:
            # Get staging model
            staging_model = self.mlflow_client.get_latest_versions(
                name="fraud_detection",
                stages=["Staging"]
            )[0]
            
            # Load model
            model = mlflow.pyfunc.load_model(f"models:/fraud_detection/{staging_model.version}")
            
            # Get features and target
            X = self.reference_data.drop('Class', axis=1)
            y = self.reference_data['Class']
            
            # Get predictions
            predictions = model.predict(X)
            
            # Calculate AUC
            auc_score = roc_auc_score(y, predictions)
            
            # Calculate drift
            drift_report = Report(metrics=[DataDriftMetric()])
            drift_report.run(
                reference_data=pd.DataFrame({'prediction': y}),
                current_data=pd.DataFrame({'prediction': predictions})
            )
            
            drift_score = drift_report.metrics[0].result.drift_score
            
            # Add performance check against base threshold
            if auc_score < self.performance_threshold:
                raise HTTPException(
                    status_code=422,
                    detail=f"Base performance {auc_score:.3f} below minimum threshold {self.performance_threshold}"
                )
            
            return {
                'auc': auc_score,
                'drift_score': drift_score,
                'performance_threshold': self.performance_threshold,
                'drift_threshold': self.drift_threshold
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to evaluate model: {str(e)}"
            )

monitor = ModelMonitor()

@app.post("/api/v1/check_model")
async def check_model(request: Request):
    """Endpoint called by Flagger before canary deployment"""
    MODEL_CHECK_COUNTER.inc()
    
    try:
        # Get threshold from request metadata
        webhook_data = await request.json()
        threshold = float(webhook_data.get('metadata', {}).get('threshold', 1.1))
        
        with CHECK_DURATION.time():
            # Get metrics
            ref_metrics = monitor.get_reference_metrics()
            current_metrics = await monitor.evaluate_current_model()
            
            # Check performance degradation
            degradation = (ref_metrics['auc'] - current_metrics['auc']) / ref_metrics['auc']
            
            if degradation > threshold - 1:  # Convert 1.1 to 0.1
                MODEL_FAILURES.inc()
                raise HTTPException(
                    status_code=422,
                    detail={
                        "status": "failed",
                        "message": f"Performance degradation {degradation:.2%} exceeds threshold"
                    }
                )
            
            # Check drift
            if current_metrics['drift_score'] > monitor.drift_threshold:
                MODEL_FAILURES.inc()
                raise HTTPException(
                    status_code=422,
                    detail={
                        "status": "failed",
                        "message": f"Drift score {current_metrics['drift_score']:.3f} exceeds threshold"
                    }
                )
                
            return {
                "status": "success",
                "statusCode": 200,
                "message": "Model performing within thresholds",
                "metrics": {
                    "auc_score": current_metrics['auc'],
                    "drift_score": current_metrics['drift_score']
                }
            }
            
    except Exception as e:
        MODEL_FAILURES.inc()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "message": f"Error during model check: {str(e)}"
            }
        )

# Add environment variables to metrics output
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint with configuration"""
    from prometheus_client import Gauge
    
    config_drift = Gauge('model_drift_threshold', 'Configured drift threshold')
    config_perf = Gauge('model_performance_threshold', 'Configured performance threshold')
    
    config_drift.set(monitor.drift_threshold)
    config_perf.set(monitor.performance_threshold)
    
    return generate_latest()
