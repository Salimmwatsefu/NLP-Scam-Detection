import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "models", "demographic_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_demographic_models(X: pd.DataFrame, y: pd.Series, demographic_labels: pd.Series) -> Dict:
    """
    Train XGBoost and Logistic Regression models with demographic features
    
    Args:
        X: Feature matrix
        y: Target labels
        demographic_labels: Demographic labels (youth, low_income, general)
        
    Returns:
        Dictionary containing trained models and their metrics
    """
    logger.info("Training demographic-aware models...")
    
    # Add demographic weights as features
    demo_dummies = pd.get_dummies(demographic_labels, prefix='demo')
    demo_weights = {
        'youth': 1.2,      # Higher weight for youth-targeted scams
        'low_income': 1.1, # Moderate weight for low-income targeted scams
        'general': 1.0     # Base weight for general scams
    }
    
    # Apply weights to demographic features
    for demo in demo_weights:
        if f'demo_{demo}' in demo_dummies.columns:
            demo_dummies[f'demo_{demo}'] *= demo_weights[demo]
    
    # Combine with original features
    X_with_demo = pd.concat([X, demo_dummies], axis=1)
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        random_state=42
    )
    xgb_model.fit(X_with_demo, y)
    
    # Train Logistic Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_with_demo)
    
    lr_model = LogisticRegression(
        multi_class='multinomial',
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_scaled, y)
    
    # Calculate metrics
    xgb_pred = xgb_model.predict(X_with_demo)
    lr_pred = lr_model.predict(X_scaled)
    
    metrics = {
        'xgboost': classification_report(y, xgb_pred, output_dict=True),
        'logistic_regression': classification_report(y, lr_pred, output_dict=True)
    }
    
    # Save models and scaler
    with open(os.path.join(OUTPUT_DIR, 'xgboost_demographic.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)
    
    with open(os.path.join(OUTPUT_DIR, 'lr_demographic.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)
    
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    
    logger.info(f"Models and metrics saved to {OUTPUT_DIR}")
    
    return {
        'models': {
            'xgboost': xgb_model,
            'logistic_regression': lr_model,
            'scaler': scaler
        },
        'metrics': metrics
    } 