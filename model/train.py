import os
import mlflow
import joblib
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from pathlib import Path

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")


# Add seed to for reproducibility
np.random.seed(42)

# Enable mlflow autologging
mlflow.xgboost.autolog()
mlflow.set_tracking_uri(os.getenv("MLFLOW_SERVER"))

# read fraud data
DATA_PATH = Path(__file__).resolve().parents[1]/"data"/"credit-card.csv"
df = pd.read_csv(DATA_PATH)

# Data Inspection
logger.info(f"Dataset shape: {df.shape}")
logger.info(f"Fraud Cases: {df['Class'].sum()} / {len(df)}")

# Shuffle dataset
df = shuffle(df, random_state=42)

# Split into features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Stratified train-test split (to ensure equal distribution of fraud cases)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

mlflow.set_experiment("fraud-detection")
with mlflow.start_run(run_name="xgboost-v1-credit-card"):
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    auc = roc_auc_score(y_val, y_pred_proba)
    logger.info(f"\n🔍 ROC AUC Score: {auc:.4f}")

    logger.info("\n📊 Classification Report: ")
    logger.info(classification_report(y_val, y_pred, digits=4))

    # log experiment
    mlflow.log_metric("auc", roc_auc_score(y_val, y_pred_proba))
    mlflow.log_metric("recall", recall_score(y_val, y_pred))
    mlflow.set_tag("model_type", "fraud_xgboost_v1")

    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_val, y_pred=y_pred)
    plt.savefig("confusion-matrix.png")
    mlflow.log_artifact("confusion-matrix.png")

# model output path
MODEL_OUTPUT_PATH = Path(__file__).resolve().parents[0]/"model.pkl"

# Save the trained model
joblib.dump(model, MODEL_OUTPUT_PATH)
logger.info(f"\n✅ Model saved to: {MODEL_OUTPUT_PATH}")