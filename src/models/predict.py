import joblib
import numpy as np
from pathlib import Path
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import re
import spacy
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path(__file__).parent.parent.parent / "outputs" / "models"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"

# Initialize models, tokenizer, vectorizer
models = {}
tokenizer = None
bert_model = None
tfidf_vectorizer = None
nlp = spacy.load("en_core_web_sm")
DetectorFactory.seed = 0

def load_config():
    """Load configuration from config.yaml."""
    try:
        with open(CONFIG_PATH, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {CONFIG_PATH}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file {CONFIG_PATH} not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise

def preprocess_text(text, config):
    """Preprocess text and extract features, matching preprocessing script."""
    # Handle invalid inputs
    if not isinstance(text, str) or pd.isna(text) or text.strip() == "":
        logger.warning(f"Invalid input text: {text}. Returning default 'empty'.")
        text = "empty"
    
    original_text = text
    text = text.lower()
    text = re.sub(r'\\n|\n+|\r|\t|\f|\v', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+|[A-Za-z0-9.-]+\.(com|org|net)\b', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove names
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            text = text.replace(ent.text, '')

    text = re.sub(r'\s+', ' ', text.strip())

    # Lemmatize + clean tokens
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha and len(token.text) > 2]
    cleaned_text = " ".join(tokens)

    # Handle empty cleaned_text
    if not cleaned_text.strip():
        logger.warning(f"Cleaned text is empty for input: {original_text}. Using 'empty'.")
        cleaned_text = "empty"

    # Extract features
    cap_words = len([word for word in original_text.split() if any(c.isupper() for c in word) and word.isalpha()])
    has_caps = cap_words > 0
    has_phone = bool(re.search(config["preprocessing"]["phone_regex"], original_text))

    try:
        lang = detect(original_text)
        is_kiswahili_sheng = lang in ['sw', 'so']
    except:
        is_kiswahili_sheng = False

    has_sheng = any(keyword in original_text.lower() for keyword in config["preprocessing"]["sheng_keywords"])
    is_kiswahili_sheng = is_kiswahili_sheng or has_sheng

    return cleaned_text, has_caps, has_phone, is_kiswahili_sheng

def load_models():
    """Load all models and vectorizer into memory."""
    global models, tokenizer, bert_model, tfidf_vectorizer
    
    try:
        # Load configuration
        config = load_config()
        
        # Load TF-IDF vectorizer
        tfidf_vectorizer_path = Path(config["features"]["tfidf_vectorizer_dir"]) / f"tfidf_vectorizer_{config['run']['tag']}.pkl"
        if tfidf_vectorizer_path.exists():
            tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
            logger.info(f"Loaded TF-IDF vectorizer from {tfidf_vectorizer_path}")
        else:
            logger.warning(f"TF-IDF vectorizer not found at {tfidf_vectorizer_path}")
        
        # Load XGBoost model
        xgb_path = MODEL_DIR / "xgboost_model.joblib"
        if xgb_path.exists():
            models['XGBoost'] = joblib.load(xgb_path)
            logger.info(f"Loaded XGBoost model from {xgb_path}")
            
        # Load Logistic Regression model
        lr_path = MODEL_DIR / "logistic_regression_model.joblib"
        if lr_path.exists():
            models['Logistic Regression'] = joblib.load(lr_path)
            logger.info(f"Loaded Logistic Regression model from {lr_path}")
            
        # Load BERT model and tokenizer
        bert_path = MODEL_DIR / "bert_model"
        if bert_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(bert_path)
            bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
            models['BERT'] = bert_model
            logger.info(f"Loaded BERT model and tokenizer from {bert_path}")
            
        if not models:
            logger.error("No models were loaded successfully")
            raise ValueError("No models were loaded")
            
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
            
        # Load configuration for preprocessing
        config = load_config()
        
        # Preprocess the message
        cleaned_text, has_caps, has_phone, is_kiswahili_sheng = preprocess_text(message, config)
        
        # Get prediction based on model type
        if model_type == 'BERT':
            # BERT prediction
            inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=config["features"]["bert_max_length"])
            with torch.no_grad():
                outputs = bert_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1).numpy()[0]
            prediction_idx = np.argmax(probabilities)
            probability = probabilities[prediction_idx]
            
            # Map prediction index to binary label
            label_map = {0: 'legit', 1: 'scam'}
            prediction = label_map[prediction_idx]
            
        else:
            # Traditional ML models (XGBoost, Logistic Regression)
            if tfidf_vectorizer is None:
                raise ValueError("TF-IDF vectorizer not loaded")
                
            # Transform text to TF-IDF features
            X_tfidf = tfidf_vectorizer.transform([cleaned_text]).toarray()
            # Combine with binary features
            binary_features = np.array([[has_caps, has_phone, is_kiswahili_sheng]], dtype=int)
            X_combined = np.hstack([X_tfidf, binary_features])
            
            model = models[model_type]
            probabilities = model.predict_proba(X_combined)[0]
            prediction_idx = np.argmax(probabilities)
            probability = probabilities[prediction_idx]
            
            # Map prediction index to binary label
            label_map = {0: 'legit', 1: 'scam'}
            prediction = label_map[prediction_idx]
            
        return prediction, probability
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

# Initialize models when module is imported
logger.info("Initializing model loading...")
load_models()