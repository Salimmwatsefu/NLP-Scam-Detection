import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from sklearn.model_selection import train_test_split
from models.train import train_model
from utils.preprocess import preprocess_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path(__file__).parent.parent.parent / "outputs" / "models"

def prepare_training_data(feedback_data):
    """
    Prepare training data from feedback.
    
    Args:
        feedback_data (list): List of feedback dictionaries
        
    Returns:
        tuple: (X, y) training data and labels
    """
    messages = []
    labels = []
    
    for feedback in feedback_data:
        messages.append(feedback["message"])
        # Map risk levels to numeric labels
        risk_level = feedback["correct_label"]
        if risk_level == "Low-risk":
            label = 0
        elif risk_level == "Moderate-risk":
            label = 1
        else:  # High-risk
            label = 2
        labels.append(label)
    
    # Preprocess messages
    X = [preprocess_text(msg) for msg in messages]
    y = np.array(labels)
    
    return X, y

def update_model(feedback_data, model_type="XGBoost"):
    """
    Update model with new feedback data.
    
    Args:
        feedback_data (list): List of feedback dictionaries
        model_type (str): Type of model to update
        
    Returns:
        bool: True if update successful, False otherwise
    """
    try:
        # Load original training data
        data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "scam_preprocessed.csv"
        df = pd.read_csv(data_path)
        
        # Prepare feedback data
        X_feedback, y_feedback = prepare_training_data(feedback_data)
        
        # Combine with original data
        X_original = df['message_content'].apply(preprocess_text).values
        y_original = df['label'].values
        
        X_combined = np.concatenate([X_original, X_feedback])
        y_combined = np.concatenate([y_original, y_feedback])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42
        )
        
        # Train new model
        model = train_model(X_train, y_train, model_type=model_type)
        
        # Save updated model
        model_path = MODEL_DIR / f"{model_type.lower()}_updated.joblib"
        joblib.dump(model, model_path)
        
        logger.info(f"Model updated successfully with {len(feedback_data)} new samples")
        return True
        
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        return False

def get_model_performance(model_type="XGBoost"):
    """
    Get performance metrics for the updated model.
    
    Args:
        model_type (str): Type of model to evaluate
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    try:
        # Load test data
        data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "scam_preprocessed.csv"
        df = pd.read_csv(data_path)
        
        # Load updated model
        model_path = MODEL_DIR / f"{model_type.lower()}_updated.joblib"
        if not model_path.exists():
            return None
            
        model = joblib.load(model_path)
        
        # Prepare test data
        X_test = df['message_content'].apply(preprocess_text).values
        y_test = df['label'].values
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        return None 