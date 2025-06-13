import joblib
import numpy as np
from pathlib import Path
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path(__file__).parent.parent.parent / "outputs" / "models"

# Initialize models dictionary
models = {}
tokenizer = None
bert_model = None

def load_models():
    """Load all models into memory."""
    global models, tokenizer, bert_model
    
    try:
        # Load XGBoost model
        xgb_path = MODEL_DIR / "xgboost_model.joblib"
        if xgb_path.exists():
            models['XGBoost'] = joblib.load(xgb_path)
            
        # Load Logistic Regression model
        lr_path = MODEL_DIR / "logistic_regression_model.joblib"
        if lr_path.exists():
            models['Logistic Regression'] = joblib.load(lr_path)
            
        # Load BERT model and tokenizer
        bert_path = MODEL_DIR / "bert_model"
        if bert_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(bert_path)
            bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
            models['BERT'] = bert_model
            
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def predict_message(message, model_type="XGBoost"):
    """
    Predict whether a message is a scam.
    
    Args:
        message (str): The message to predict
        model_type (str): Type of model to use ('XGBoost', 'Logistic Regression', or 'BERT')
        
    Returns:
        tuple: (prediction, probability)
    """
    try:
        if not models:
            load_models()
            
        if model_type not in models:
            raise ValueError(f"Model type {model_type} not found")
            
        # Get prediction based on model type
        if model_type == 'BERT':
            # BERT prediction
            inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1).numpy()[0]
            prediction_idx = np.argmax(probabilities)
            probability = probabilities[prediction_idx]
            
            # Map prediction index to label
            label_map = {0: 'legit', 1: 'moderate_scam', 2: 'high_scam'}
            prediction = label_map[prediction_idx]
            
        else:
            # Traditional ML models (XGBoost, Logistic Regression)
            model = models[model_type]
            # Assuming the model expects preprocessed text
            # You might need to add preprocessing steps here
            probabilities = model.predict_proba([message])[0]
            prediction_idx = np.argmax(probabilities)
            probability = probabilities[prediction_idx]
            
            # Map prediction index to label
            label_map = {0: 'legit', 1: 'moderate_scam', 2: 'high_scam'}
            prediction = label_map[prediction_idx]
            
        return prediction, probability
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

# Initialize models when module is imported
logger.info("Initializing model loading...")
load_models() 