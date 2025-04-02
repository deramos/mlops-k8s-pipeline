import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils import shuffle
from pathlib import Path

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")


# Add seed to for reproducibility
np.random.seed(42)

# Enable mlflow autologging
mlflow.xgboost.autolog()

DATA_PATH = Path(__file__).resolve().parents[1]/"data"/"credit-card.csv"
df = pd.read_csv(DATA_PATH)

# Data Inspection
logger.info(f"Dataset shape: {df.shape}")
logger.info(f"Fraud Percentage: {df['Class'].sum()} / {len(df)}")
