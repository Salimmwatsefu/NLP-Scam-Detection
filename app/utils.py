import os
import pandas as pd
import numpy as np
import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import langid
from scipy.sparse import hstack
import logging
import shap
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize spaCy and NLTK
nlp = spacy.load("en_core_web_sm")
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Define directories
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(WORKSPACE_ROOT, '..', 'outputs', 'models')
DATA_DIR = os.path.join(WORKSPACE_ROOT, '..', 'data')
FEEDBACK_DIR = os.path.join(DATA_DIR, 'feedback')
TFIDF_DIR = os.path.join(WORKSPACE_ROOT, '..', 'data', 'features', 'tfidf')
DATASET_PATH = '/home/sjet/iwazolab/NLP-Scam-Detection/data/processed/scam_preprocessed.csv'

# Create necessary directories
for directory in [MODEL_DIR, DATA_DIR, FEEDBACK_DIR, TFIDF_DIR]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Created/verified directory: {directory}")

# Verify feedback directory is writable
try:
    feedback_test_file = os.path.join(FEEDBACK_DIR, 'test.txt')
    with open(feedback_test_file, 'w') as f:
        f.write('test')
    os.remove(feedback_test_file)
    logger.info(f"Feedback directory {FEEDBACK_DIR} is writable")
except Exception as e:
    logger.error(f"Error writing to feedback directory: {str(e)}")

# Model file names
LR_MODEL_FILE = 'logistic_regression_model_scam_run_2025-07-04_03-54.joblib'
XGB_MODEL_FILE = 'xgboost_model_scam_run_2025-07-04_03-54.joblib'
TFIDF_FILE = 'tfidf_vectorizer_scam_run_2025-07-04_03-54.pkl'

# Load models
def load_models():
    try:
        lr_model_path = os.path.join(MODEL_DIR, LR_MODEL_FILE)
        xgb_model_path = os.path.join(MODEL_DIR, XGB_MODEL_FILE)
        tfidf_vectorizer_path = os.path.join(TFIDF_DIR, TFIDF_FILE)
        
        for path in [lr_model_path, xgb_model_path, tfidf_vectorizer_path]:
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return None, None, None
        
        logger.info("Loading Logistic Regression model...")
        lr_model = joblib.load(lr_model_path)
        logger.info("Logistic Regression model loaded successfully")
        
        logger.info("Loading XGBoost model...")
        xgb_model = joblib.load(xgb_model_path)
        logger.info("XGBoost model loaded successfully")
        
        logger.info("Loading TF-IDF vectorizer...")
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        logger.info("TF-IDF vectorizer loaded successfully")
        
        return lr_model, xgb_model, tfidf_vectorizer
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return None, None, None

# Preprocessing function
def preprocess_text(text):
    raw_text = str(text)
    raw_tokens = word_tokenize(raw_text.lower())
    text = raw_text.lower()
    text = re.sub(r'\\n|\n+|\r|\t|\f|\v', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'http\S+|www\S+|https\S+|[A-Za-z0-9.-]+\.(com|org|net|co\.ke)\b', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
    text = re.sub(r'\+\d{10,12}\b|\+\d{1,3}-\d{3}-\d{3}-\d{4}\b|\b(?:\+?254|0)(7\d{8}|11\d{7})\b', '', text)
    text = re.sub(r'\b[A-Z0-9]{10}\b|\bconfirmed\b|\bcompleted\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s]', ' ', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha and len(token.text) > 2]
    cleaned_text = " ".join(tokens)
    return raw_text, raw_tokens, tokens, cleaned_text

# Extract additional features
def extract_features(raw_text):
    tokens = word_tokenize(raw_text)
    has_caps = sum(1 for word in tokens if word.isupper() and len(word) > 1) >= 2
    phone_pattern = r'\+\d{10,12}\b|\+\d{1,3}-\d{3}-\d{3}-\d{4}\b|\b(?:\+?254|0)(7\d{8}|11\d{7})\b'
    has_phone = len(re.findall(phone_pattern, raw_text)) > 0
    lang, _ = langid.classify(raw_text)
    is_kiswahili_sheng = lang in ['sw', 'mixed']
    return {
        'has_caps': has_caps,
        'has_phone': has_phone,
        'is_kiswahili_sheng': is_kiswahili_sheng
    }

# Detect scam indicators
def detect_scam_indicators(raw_text, raw_tokens):
    indicators = []
    raw_text_lower = raw_text.lower()
    
    caps_count = sum(1 for word in word_tokenize(raw_text) if word.isupper() and len(word) > 1)
    if caps_count >= 2:
        indicators.append({
            'icon': '🚨',
            'type': 'Aggressive Formatting',
            'detail': f'Found {caps_count} words in ALL CAPS - scammers often use this to create urgency',
            'severity': 'medium'
        })

    links = re.findall(r'http\S+|www\S+|https\S+|[A-Za-z0-9.-]+\.(com|org|net|co\.ke)\b', raw_text)
    if links:
        indicators.append({
            'icon': '🔗',
            'type': 'Suspicious Link',
            'detail': f'Contains link(s): {", ".join(links)}. Be cautious of unexpected URLs',
            'severity': 'high'
        })

    money_keywords = ['ksh', 'cash', 'money', 'mpesa', 'pay', 'send', 'receive', 'account', 'bank', 'transaction']
    money_matches = [word for word in money_keywords if word in raw_text_lower]
    if money_matches:
        indicators.append({
            'icon': '💰',
            'type': 'Financial Bait',
            'detail': f'Contains financial terms: {", ".join(money_matches)}',
            'severity': 'high'
        })

    betting_keywords = ['spin', 'win', 'bonus', 'play', 'bet', 'jackpot', 'odds', 'stake', 'gamble']
    betting_matches = [word for word in betting_keywords if word in raw_text_lower]
    if betting_matches:
        indicators.append({
            'icon': '🎰',
            'type': 'Betting Lure',
            'detail': f'Contains betting-related terms: {", ".join(betting_matches)}',
            'severity': 'high'
        })

    urgency_keywords = ['urgent', 'hurry', 'quick', 'fast', 'now', 'today', 'limited', 'expire', 'immediate']
    urgency_matches = [word for word in urgency_keywords if word in raw_text_lower]
    if urgency_matches:
        indicators.append({
            'icon': '⚡',
            'type': 'Urgency Tactics',
            'detail': f'Uses urgency-creating words: {", ".join(urgency_matches)}',
            'severity': 'medium'
        })

    sheng_keywords = ['yako', 'mambo', 'poa', 'sasa', 'bro', 'fam', 'chap', 'dame']
    sheng_matches = [word for word in sheng_keywords if word in raw_text_lower]
    if sheng_matches:
        indicators.append({
            'icon': '🗣️',
            'type': 'Informal Language',
            'detail': f'Uses casual/Sheng phrases: {", ".join(sheng_matches)}',
            'severity': 'low'
        })

    personal_info_patterns = [
        (r'\b(verify|confirm)\s+.*(account|details|identity)\b', 'account verification'),
        (r'\b(enter|send|share|provide).*(pin|password|id)\b', 'credentials'),
        (r'\blogin\s+.*\b(account|details)\b', 'login details'),
        (r'\b(update|verify)\s+.*\b(kyc|details|information)\b', 'personal details'),
        (r'\b(send|share|enter)\s+.*\b(mpesa|m-pesa)\s+.*\b(pin)\b', 'M-PESA PIN')
    ]
    
    found_patterns = []
    for pattern, info_type in personal_info_patterns:
        if re.search(pattern, raw_text_lower):
            found_patterns.append(info_type)
    
    if found_patterns:
        indicators.append({
            'icon': '🔒',
            'type': 'Personal Info Request',
            'detail': f'Requests sensitive information: {", ".join(found_patterns)}',
            'severity': 'high'
        })

    return indicators

# Language mixing detection
def detect_language_mixing(message):
    """
    Analyze message for Swahili-English code-mixing
    Returns: dict with language percentages, word-level tags, and confidence
    """
    swahili_words = [
        'umeshinda', 'hongera', 'kupata', 'karibu', 'tumia', 'yako', 'kwa', 'na', 'ya',
        'leo', 'sasa', 'mambo', 'poa', 'zawadi', 'namba', 'hapa', 'pesa', 'akaunti',
        'bila', 'haraka', ' bure', 'tuma', 'piga', 'simu', 'jaza'  # Added common Swahili/Sheng terms
    ]
    english_words = [
        'winner', 'congratulations', 'service', 'balance', 'check', 'click', 'here', 'claim',
        'account', 'send', 'receive', 'urgent', 'offer', 'prize', 'win', 'verify', 'update',
        'login', 'password', 'details', 'money', 'cash', 'link', 'today'  # Added common English terms
    ]
    
    words = message.lower().split()
    word_tags = []
    sw_count = 0
    en_count = 0
    
    for word in words:
        is_swahili = any(sw in word for sw in swahili_words)
        is_english = any(en in word for en in english_words)
        if is_swahili and not is_english:
            word_tags.append((word, 'Swahili'))
            sw_count += 1
        elif is_english and not is_swahili:
            word_tags.append((word, 'English'))
            en_count += 1
        else:
            word_tags.append((word, 'Mixed/Other'))
    
    total_words = len(words)
    sw_percentage = (sw_count / total_words) * 100 if total_words > 0 else 0
    en_percentage = (en_count / total_words) * 100 if total_words > 0 else 0
    mixed_percentage = 100 - sw_percentage - en_percentage
    code_mixing_intensity = min(sw_percentage, en_percentage) * 2
    
    # Calculate linguistic confidence based on word coverage
    known_words = sw_count + en_count
    linguistic_confidence = (known_words / total_words) * 100 if total_words > 0 else 0
    
    return {
        'swahili_pct': sw_percentage,
        'english_pct': en_percentage,
        'mixed_pct': mixed_percentage,
        'code_mixing_intensity': code_mixing_intensity,
        'word_tags': word_tags,
        'linguistic_confidence': linguistic_confidence
    }

# Prediction function
def predict(text, selected_model, lr_model, xgb_model, tfidf_vectorizer):
    if lr_model is None or xgb_model is None or tfidf_vectorizer is None:
        logger.error("Models not loaded. Cannot perform prediction.")
        return "Error: Models not loaded", 0.0, [], "", {}, [], [], {}
    
    raw_text, raw_tokens, tokens, cleaned_text = preprocess_text(text)
    features = extract_features(raw_text)
    indicators = detect_scam_indicators(raw_text, raw_tokens)
    language_analysis = detect_language_mixing(raw_text)
    labels = ['legit', 'scam']
    
    if not cleaned_text.strip():
        return "Invalid Input", 0.0, tokens, cleaned_text, features, indicators, [], language_analysis
    
    try:
        tfidf_features = tfidf_vectorizer.transform([cleaned_text])
        extra_features_np = np.array([[features['has_caps'], features['has_phone'], features['is_kiswahili_sheng']]], dtype=float)
        combined_features = hstack([tfidf_features, extra_features_np]).toarray()
        
        label_idx = 0
        confidence = 0.0
        top_features = []
        
        if selected_model == "Logistic Regression":
            lr_probs = lr_model.predict_proba(combined_features)
            label_idx = np.argmax(lr_probs, axis=1)[0]
            confidence = lr_probs[0][label_idx]
            top_features = ["Feature importance not available for Logistic Regression"]
        
        elif selected_model == "XGBoost":
            xgb_probs = xgb_model.predict_proba(combined_features)
            label_idx = np.argmax(xgb_probs, axis=1)[0]
            confidence = xgb_probs[0][label_idx]
            try:
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer.shap_values(combined_features)
                feature_names = tfidf_vectorizer.get_feature_names_out().tolist() + ['has_caps', 'has_phone', 'is_kiswahili_sheng']
                
                class_idx = label_idx
                if isinstance(shap_values, list):
                    class_shap_values = np.abs(shap_values[class_idx][0])
                else:
                    class_shap_values = np.abs(shap_values[0])
                
                feature_importance = list(zip(feature_names, class_shap_values))
                sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
                top_features = [(feature, float(importance)) for feature, importance in sorted_features[:10]]
            
            except Exception as e:
                logger.error(f"Error calculating SHAP values: {str(e)}")
                top_features = ["Error calculating feature importance"]
        
        return labels[label_idx], confidence, tokens, cleaned_text, features, indicators, top_features, language_analysis
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return "Error in prediction", 0.0, tokens, cleaned_text, features, indicators, [], language_analysis

# Save user feedback
def save_feedback(text, prediction, user_correction, selected_model):
    try:
        feedback_data = {
            'timestamp': pd.Timestamp.now(),
            'message': text,
            'original_prediction': prediction,
            'correct_label': user_correction,
            'model_type': selected_model
        }
        feedback_df = pd.DataFrame([feedback_data])
        feedback_path = os.path.join(FEEDBACK_DIR, 'user_feedback.csv')
        if os.path.exists(feedback_path):
            feedback_df.to_csv(feedback_path, mode='a', header=False, index=False)
        else:
            feedback_df.to_csv(feedback_path, index=False)
        logger.info(f"Feedback saved successfully to {feedback_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        return False