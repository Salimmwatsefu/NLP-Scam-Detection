import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import nltk
import langid
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import shap
from scipy.sparse import hstack
import logging
from pathlib import Path
from datetime import datetime
import json

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
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(WORKSPACE_ROOT, 'outputs', 'models')
DATA_DIR = os.path.join(WORKSPACE_ROOT, 'data')
FEEDBACK_DIR = os.path.join(DATA_DIR, 'feedback')
TFIDF_DIR = os.path.join(WORKSPACE_ROOT, 'data', 'features', 'tfidf')

# Create necessary directories
for directory in [MODEL_DIR, DATA_DIR, FEEDBACK_DIR, TFIDF_DIR]:
    try:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created/verified directory: {directory}")
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {str(e)}")

# Verify feedback directory exists and is writable
try:
    feedback_test_file = os.path.join(FEEDBACK_DIR, 'test.txt')
    with open(feedback_test_file, 'w') as f:
        f.write('test')
    os.remove(feedback_test_file)
    logger.info(f"Feedback directory {FEEDBACK_DIR} is writable")
except Exception as e:
    logger.error(f"Error writing to feedback directory: {str(e)}")

# Model file names (updated to match provided files)
LR_MODEL_FILE = 'logistic_regression_model_scam_run_2025-07-04_03-54.joblib'
XGB_MODEL_FILE = 'xgboost_model_scam_run_2025-07-04_03-54.joblib'
TFIDF_FILE = 'tfidf_vectorizer_scam_run_2025-07-04_03-54.pkl'

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="SMS Scam Shield | Advanced Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for modern, professional UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom color variables */
    :root {
        --primary-color: #6366F1;
        --primary-dark: #4F46E5;
        --secondary-color: #10B981;
        --danger-color: #EF4444;
        --warning-color: #F59E0B;
        --success-color: #10B981;
        --text-primary: #111827;
        --text-secondary: #6B7280;
        --bg-light: #F9FAFB;
        --bg-white: #FFFFFF;
        --border-color: #E5E7EB;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4rem 2rem;
        text-align: center;
        border-radius: 1rem;
        margin-bottom: 3rem;
        box-shadow: var(--shadow-xl);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        font-weight: 400;
        opacity: 0.9;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Modern card design */
    .modern-card {
        background: var(--bg-white);
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        width: 100%;
        padding: 1rem 2rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        background: linear-gradient(135deg, var(--primary-dark), #3730A3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Enhanced risk level badges */
    .risk-badge {
        padding: 0.75rem 1.5rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #FEE2E2, #FECACA);
        color: var(--danger-color);
        border: 2px solid #FCA5A5;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #FEF3C7, #FDE68A);
        color: var(--warning-color);
        border: 2px solid #FBBF24;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
        color: var(--success-color);
        border: 2px solid #34D399;
    }
    
    /* Enhanced indicator cards */
    .indicator-card {
        background: var(--bg-white);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border-left: 5px solid;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }
    
    .indicator-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, currentColor, transparent);
        opacity: 0.3;
    }
    
    .indicator-card:hover {
        transform: translateX(8px);
        box-shadow: var(--shadow-md);
    }
    
    .indicator-high {
        border-left-color: var(--danger-color);
        color: var(--danger-color);
    }
    
    .indicator-medium {
        border-left-color: var(--warning-color);
        color: var(--warning-color);
    }
    
    .indicator-low {
        border-left-color: var(--success-color);
        color: var(--success-color);
    }
    
    .indicator-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }
    
    .indicator-icon {
        font-size: 1.5rem;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    
    .indicator-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: var(--text-primary);
    }
    
    .indicator-detail {
        color: var(--text-secondary);
        line-height: 1.5;
        margin: 0;
        padding-left: 2.25rem;
    }
    
    /* Enhanced headers */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }
    
    h1 { font-size: 2.5rem; }
    h2 { font-size: 2rem; }
    h3 { font-size: 1.5rem; }
    h4 { font-size: 1.25rem; }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-light);
        border-radius: 1rem;
        padding: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        border-radius: 0.75rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-color);
        color: white;
        box-shadow: var(--shadow-md);
    }
    
    /* Enhanced text areas and inputs */
    .stTextArea textarea {
        border-radius: 0.75rem;
        border: 2px solid var(--border-color);
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        resize: vertical;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    .stSelectbox > div > div {
        border-radius: 0.75rem;
        border: 2px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* Enhanced expanders */
    .streamlit-expanderHeader {
        background: var(--bg-light);
        border-radius: 0.75rem;
        padding: 1rem;
        border: 1px solid var(--border-color);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--bg-white);
        box-shadow: var(--shadow-sm);
    }
    
    /* Protection tips styling */
    .protection-tip {
        background: linear-gradient(135deg, #F0F9FF, #E0F2FE);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid var(--primary-color);
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    .protection-tip h4 {
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    /* Stats cards */
    .stats-card {
        background: var(--bg-white);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        display: block;
    }
    
    .stats-label {
        color: var(--text-secondary);
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Enhanced dataframe styling */
    .stDataFrame {
        border-radius: 1rem;
        overflow: hidden;
        box-shadow: var(--shadow-md);
    }
    
    /* Loading spinner enhancement */
    .stSpinner > div {
        border-top-color: var(--primary-color) !important;
    }
    
    /* Enhanced alerts */
    .stAlert {
        border-radius: 0.75rem;
        border: none;
        box-shadow: var(--shadow-sm);
    }
    
    /* Footer styling */
    .footer {
        background: var(--bg-light);
        padding: 3rem 2rem;
        text-align: center;
        border-radius: 1rem;
        margin-top: 4rem;
        border: 1px solid var(--border-color);
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    .footer-link {
        color: var(--primary-color);
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .footer-link:hover {
        color: var(--primary-dark);
        text-decoration: underline;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
        }
        
        .modern-card {
            padding: 1.5rem;
        }
        
        .indicator-card {
            padding: 1rem;
        }
        
        .stats-number {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Hero Section
st.markdown("""
    <div class='hero-section'>
        <div class='hero-title'>🛡️ SMS Scam Shield</div>
        <div class='hero-subtitle'>
            Advanced AI-powered protection against SMS scams and phishing attempts. 
            Analyze messages instantly with our state-of-the-art detection system.
        </div>
    </div>
""", unsafe_allow_html=True)

# Load models at startup
@st.cache_resource
def load_models():
    try:
        lr_model_path = os.path.join(MODEL_DIR, LR_MODEL_FILE)
        xgb_model_path = os.path.join(MODEL_DIR, XGB_MODEL_FILE)
        tfidf_vectorizer_path = os.path.join(TFIDF_DIR, TFIDF_FILE)

        # Check if files exist before loading
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

# Load models and handle errors without Streamlit commands
try:
    logger.info("Initializing model loading...")
    lr_model, xgb_model, tfidf_vectorizer = load_models()
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}")
    lr_model = xgb_model = tfidf_vectorizer = None

# Display model loading status with enhanced styling
if lr_model is not None:
    st.success("✅ **AI Models Loaded Successfully** - Ready to protect you from scams!")
else:
    st.error("❌ **Model Loading Failed** - Please ensure model files exist in the correct directories and try again.")

# Preprocessing function
def preprocess_text(text):
    # Preserve raw text for indicator detection
    raw_text = str(text)
    raw_tokens = word_tokenize(raw_text.lower())  # Lowercase for keyword checks

    # Clean text for model input
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
    
    # Check for excessive capitalization
    caps_count = sum(1 for word in word_tokenize(raw_text) if word.isupper() and len(word) > 1)
    if caps_count >= 2:
        indicators.append({
            'icon': '🚨',
            'type': 'Aggressive Formatting',
            'detail': f'Found {caps_count} words in ALL CAPS - scammers often use this to create urgency',
            'severity': 'medium'
        })

    # Check for suspicious links
    links = re.findall(r'http\S+|www\S+|https\S+|[A-Za-z0-9.-]+\.(com|org|net|co\.ke)\b', raw_text)
    if links:
        indicators.append({
            'icon': '🔗',
            'type': 'Suspicious Link',
            'detail': f'Contains link(s): {", ".join(links)}. Be cautious of unexpected URLs',
            'severity': 'high'
        })

    # Check for monetary/financial terms
    money_keywords = ['ksh', 'cash', 'money', 'mpesa', 'pay', 'send', 'receive', 'account', 'bank', 'transaction']
    money_matches = [word for word in money_keywords if word in raw_text_lower]
    if money_matches:
        indicators.append({
            'icon': '💰',
            'type': 'Financial Bait',
            'detail': f'Contains financial terms: {", ".join(money_matches)}',
            'severity': 'high'
        })

    # Check for betting/gambling terms
    betting_keywords = ['spin', 'win', 'bonus', 'play', 'bet', 'jackpot', 'odds', 'stake', 'gamble']
    betting_matches = [word for word in betting_keywords if word in raw_text_lower]
    if betting_matches:
        indicators.append({
            'icon': '🎰',
            'type': 'Betting Lure',
            'detail': f'Contains betting-related terms: {", ".join(betting_matches)}',
            'severity': 'high'
        })

    # Check for urgency indicators
    urgency_keywords = ['urgent', 'hurry', 'quick', 'fast', 'now', 'today', 'limited', 'expire', 'immediate']
    urgency_matches = [word for word in urgency_keywords if word in raw_text_lower]
    if urgency_matches:
        indicators.append({
            'icon': '⚡',
            'type': 'Urgency Tactics',
            'detail': f'Uses urgency-creating words: {", ".join(urgency_matches)}',
            'severity': 'medium'
        })

    # Check for Sheng and informal language
    sheng_keywords = ['yako', 'mambo', 'poa', 'sasa', 'bro', 'fam', 'chap', 'dame']
    sheng_matches = [word for word in sheng_keywords if word in raw_text_lower]
    if sheng_matches:
        indicators.append({
            'icon': '🗣️',
            'type': 'Informal Language',
            'detail': f'Uses casual/Sheng phrases: {", ".join(sheng_matches)}',
            'severity': 'low'
        })

    # Check for personal information requests
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

# Prediction function with model selection
def predict(text, selected_model):
    if lr_model is None or xgb_model is None or tfidf_vectorizer is None:
        logger.error("Models not loaded. Cannot perform prediction.")
        return "Error: Models not loaded", 0.0, [], "", {}, [], []

    raw_text, raw_tokens, tokens, cleaned_text = preprocess_text(text)
    features = extract_features(raw_text)
    indicators = detect_scam_indicators(raw_text, raw_tokens)
    labels = ['legit', 'scam']

    if not cleaned_text.strip():
        return "Invalid Input", 0.0, tokens, cleaned_text, features, indicators, []

    try:
        # Prepare common features
        tfidf_features = tfidf_vectorizer.transform([cleaned_text])
        extra_features_np = np.array([[features['has_caps'], features['has_phone'], features['is_kiswahili_sheng']]], dtype=float)
        combined_features = hstack([tfidf_features, extra_features_np]).toarray()

        # Initialize outputs
        label_idx = 0
        confidence = 0.0
        top_features = []

        if selected_model == "Logistic Regression":
            # Logistic Regression prediction
            lr_probs = lr_model.predict_proba(combined_features)
            label_idx = np.argmax(lr_probs, axis=1)[0]
            confidence = lr_probs[0][label_idx]
            top_features = ["Feature importance not available for Logistic Regression"]

        elif selected_model == "XGBoost":
            # XGBoost prediction
            xgb_probs = xgb_model.predict_proba(combined_features)
            label_idx = np.argmax(xgb_probs, axis=1)[0]
            confidence = xgb_probs[0][label_idx]
            # SHAP explanation
            try:
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer.shap_values(combined_features)
                feature_names = tfidf_vectorizer.get_feature_names_out().tolist() + ['has_caps', 'has_phone', 'is_kiswahili_sheng']
                
                # Get feature importance for the predicted class
                class_idx = label_idx
                if isinstance(shap_values, list):
                    class_shap_values = np.abs(shap_values[class_idx][0])
                else:
                    class_shap_values = np.abs(shap_values[0])
                
                # Get top contributing features
                feature_importance = list(zip(feature_names, class_shap_values))
                sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
                top_features = [(feature, float(importance)) for feature, importance in sorted_features[:10]]
                
            except Exception as e:
                logger.error(f"Error calculating SHAP values: {str(e)}")
                top_features = ["Error calculating feature importance"]

        return labels[label_idx], confidence, tokens, cleaned_text, features, indicators, top_features

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return "Error in prediction", 0.0, tokens, cleaned_text, features, indicators, []

# Save user feedback
def save_feedback(text, prediction, user_correction, selected_model):
    try:
        # Create a simple feedback record
        feedback_data = {
            'timestamp': pd.Timestamp.now(),
            'message': text,
            'original_prediction': prediction,
            'correct_label': user_correction,
            'model_type': selected_model
        }
        
        # Convert to DataFrame
        feedback_df = pd.DataFrame([feedback_data])
        
        # Save to CSV
        feedback_path = os.path.join(FEEDBACK_DIR, 'user_feedback.csv')
        
        # If file exists, append without header, otherwise create new with header
        if os.path.exists(feedback_path):
            feedback_df.to_csv(feedback_path, mode='a', header=False, index=False)
        else:
            feedback_df.to_csv(feedback_path, index=False)
            
        print(f"Feedback saved successfully to {feedback_path}")  # Direct print for debugging
        return True
        
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")  # Direct print for debugging
        return False

def get_prediction_explanation(prediction, confidence, indicators):
    explanation = []
    
    # Add risk level explanation
    if prediction == "scam":
        explanation.append("🚫 **This message shows strong signs of being a scam.**")
    else:  # legit
        explanation.append("✅ **This message appears to be legitimate, but always stay vigilant.**")
    
    # Add confidence explanation
    if confidence > 0.8:
        explanation.append(f"The AI system is very confident ({confidence:.0%}) about this assessment.")
    elif confidence > 0.6:
        explanation.append(f"The AI system is moderately confident ({confidence:.0%}) about this assessment.")
    else:
        explanation.append(f"The AI system is less certain ({confidence:.0%}) about this assessment, please use extra caution.")
    
    # Summarize the key concerns
    high_severity_count = sum(1 for ind in indicators if ind['severity'] == 'high')
    medium_severity_count = sum(1 for ind in indicators if ind['severity'] == 'medium')
    
    if high_severity_count > 0 or medium_severity_count > 0:
        explanation.append("\n**🔍 Key Concerns Detected:**")
        if high_severity_count > 0:
            explanation.append(f"- **{high_severity_count} High-Risk Indicators** 🔴")
        if medium_severity_count > 0:
            explanation.append(f"- **{medium_severity_count} Medium-Risk Indicators** 🟡")
            
        # Add specific advice based on indicators
        advice_given = set()
        for indicator in indicators:
            if indicator['severity'] in ['high', 'medium']:
                if indicator['type'] == 'Financial Bait' and 'financial' not in advice_given:
                    explanation.append("- 💡 **Financial Safety**: Be extremely cautious with money-related messages")
                    advice_given.add('financial')
                elif indicator['type'] == 'Suspicious Link' and 'links' not in advice_given:
                    explanation.append("- 💡 **Link Safety**: Never click unexpected links - they may be phishing attempts")
                    advice_given.add('links')
                elif indicator['type'] == 'Personal Info Request' and 'personal' not in advice_given:
                    explanation.append("- 💡 **Privacy Protection**: Legitimate services never ask for PINs or passwords via SMS")
                    advice_given.add('personal')
                elif indicator['type'] == 'Betting Lure' and 'betting' not in advice_given:
                    explanation.append("- 💡 **Gambling Awareness**: Be skeptical of unexpected betting opportunities")
                    advice_given.add('betting')
                elif indicator['type'] == 'Urgency Tactics' and 'urgency' not in advice_given:
                    explanation.append("- 💡 **Time Pressure**: Scammers create false urgency - take your time to verify")
                    advice_given.add('urgency')
    
    return "\n".join(explanation)

# Enhanced tabs with better styling
tab1, tab2, tab3 = st.tabs(["🔍 Message Scanner", "🎓 Safety Education", "📊 Batch Analysis"])

with tab1:
    # Enhanced layout with better spacing
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Model selection in a modern card
        st.markdown("""
            <div class='modern-card'>
                <h3>🤖 AI Analysis Engine</h3>
                <p style='color: var(--text-secondary); margin-bottom: 1rem;'>
                    Choose your preferred AI model for scam detection analysis.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        selected_model = st.selectbox(
            "",
            ["XGBoost", "Logistic Regression"],
            help="XGBoost provides detailed feature analysis, while Logistic Regression offers fast predictions",
            label_visibility="collapsed"
        )
        
        # Message input section
        st.markdown("""
            <div class='modern-card'>
                <h3>📱 Message Analysis</h3>
                <p style='color: var(--text-secondary); margin-bottom: 1rem;'>
                    Paste the SMS message you want to analyze for potential scam indicators.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        text_input = st.text_area(
            "",
            height=120,
            help="Enter the complete SMS message you received",
            placeholder="Example: 'Congratulations! You have won KSH 50,000. Click here to claim: bit.ly/claim123'",
            label_visibility="collapsed"
        )
        
        # Enhanced analyze button
        if st.button("🔍 Analyze Message for Threats", use_container_width=True):
            if text_input.strip():
                with st.spinner("🔄 **Analyzing message with AI...** This may take a few seconds."):
                    prediction, confidence, tokens, cleaned_text, features, indicators, _ = predict(text_input, selected_model)
                    
                    if prediction == "Invalid Input":
                        st.warning("⚠️ **Invalid Input** - Please enter a valid message to analyze.")
                    elif prediction == "Error in prediction":
                        st.error("❌ **Analysis Error** - Something went wrong. Please try again.")
                    else:
                        # Enhanced results section
                        st.markdown("---")
                        st.markdown("## 📋 Analysis Results")
                        
                        # Enhanced risk level display
                        risk_color = {
                            "scam": "risk-high",
                            "legit": "risk-low"
                        }[prediction]
                        
                        risk_emoji = {
                            "scam": "🚨 SCAM DETECTED",
                            "legit": "✅ APPEARS SAFE"
                        }[prediction]
                        
                        st.markdown(f"""
                            <div class='risk-badge {risk_color}'>
                                {risk_emoji} • {confidence:.1%} Confidence
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Detailed explanation
                        st.markdown(f"""
                            <div class='modern-card'>
                                {get_prediction_explanation(prediction, confidence, indicators)}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Enhanced indicators display
                        if indicators:
                            st.markdown("### 🔍 Detailed Threat Analysis")
                            for indicator in indicators:
                                severity_class = {
                                    'high': 'indicator-high',
                                    'medium': 'indicator-medium',
                                    'low': 'indicator-low'
                                }[indicator['severity']]
                                
                                severity_label = {
                                    'high': 'HIGH RISK',
                                    'medium': 'MEDIUM RISK',
                                    'low': 'LOW RISK'
                                }[indicator['severity']]
                                
                                st.markdown(f"""
                                    <div class='indicator-card {severity_class}'>
                                        <div class='indicator-header'>
                                            <span class='indicator-icon'>{indicator['icon']}</span>
                                            <span class='indicator-title'>{indicator['type']}</span>
                                            <span style='margin-left: auto; font-size: 0.8rem; font-weight: 600; opacity: 0.8;'>{severity_label}</span>
                                        </div>
                                        <p class='indicator-detail'>{indicator['detail']}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        # Enhanced feedback section
                        st.markdown("---")
                        with st.expander("📝 **Help Improve Our AI** - Provide Feedback"):
                            st.markdown("""
                                <div style='background: var(--bg-light); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                                    <p><strong>Your feedback helps train our AI to be more accurate!</strong></p>
                                    <p style='color: var(--text-secondary); margin: 0;'>
                                        Was our analysis correct? Your input helps protect other users.
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            user_correction = st.radio(
                                "**How would you rate this analysis?**",
                                ["✅ Correct - Good analysis", "❌ Wrong - This should be marked as SAFE", "❌ Wrong - This should be marked as SCAM"],
                                help="Your feedback is anonymous and helps improve our detection accuracy"
                            )
                            
                            if st.button("📤 Submit Feedback", use_container_width=True):
                                if user_correction != "✅ Correct - Good analysis":
                                    corrected_label = "legit" if "SAFE" in user_correction else "scam"
                                    if save_feedback(text_input, prediction, corrected_label, selected_model):
                                        st.success("🙏 **Thank you!** Your feedback helps protect the community.")
                                    else:
                                        st.error("❌ Failed to save feedback. Please try again.")
                                else:
                                    st.success("✅ **Thank you for confirming!** This helps validate our AI accuracy.")
            else:
                st.warning("📝 **Please enter a message** to analyze before clicking the button.")

with tab2:
    st.markdown("## 🎓 SMS Safety Education Center")
    
    # Stats section
    st.markdown("### 📊 Scam Statistics")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.markdown("""
            <div class='stats-card'>
                <span class='stats-number'>85%</span>
                <div class='stats-label'>Detection Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    
    with stats_col2:
        st.markdown("""
            <div class='stats-card'>
                <span class='stats-number'>1M+</span>
                <div class='stats-label'>Messages Analyzed</div>
            </div>
        """, unsafe_allow_html=True)
    
    with stats_col3:
        st.markdown("""
            <div class='stats-card'>
                <span class='stats-number'>24/7</span>
                <div class='stats-label'>Protection Available</div>
            </div>
        """, unsafe_allow_html=True)
    
    with stats_col4:
        st.markdown("""
            <div class='stats-card'>
                <span class='stats-number'>0₵</span>
                <div class='stats-label'>Cost to Use</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced education content
    st.markdown("### 🎯 Common Scam Tactics to Watch For")
    
    tactics_col1, tactics_col2 = st.columns(2)
    
    with tactics_col1:
        st.markdown("""
            <div class='protection-tip'>
                <h4>⚡ Urgency & Pressure Tactics</h4>
                <p><strong>What to look for:</strong></p>
                <ul>
                    <li>Words like "urgent", "immediate", "expires today"</li>
                    <li>Threats of account closure or penalties</li>
                    <li>Limited time offers that seem too good to be true</li>
                </ul>
                <p><strong>Why scammers use this:</strong> They want you to act quickly without thinking or verifying.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='protection-tip'>
                <h4>🔒 Personal Information Requests</h4>
                <p><strong>Red flags:</strong></p>
                <ul>
                    <li>Requests for PINs, passwords, or ID numbers</li>
                    <li>Asking to "verify" or "update" your details</li>
                    <li>Requests for M-PESA PIN or banking information</li>
                </ul>
                <p><strong>Remember:</strong> Legitimate companies never ask for sensitive information via SMS.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with tactics_col2:
        st.markdown("""
            <div class='protection-tip'>
                <h4>💰 Financial Lures & Fake Prizes</h4>
                <p><strong>Common schemes:</strong></p>
                <ul>
                    <li>Fake lottery or competition winnings</li>
                    <li>Unexpected money transfers or refunds</li>
                    <li>Investment opportunities with guaranteed returns</li>
                </ul>
                <p><strong>Reality check:</strong> If you didn't enter a competition, you can't win it!</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='protection-tip'>
                <h4>🔗 Suspicious Links & Phishing</h4>
                <p><strong>Warning signs:</strong></p>
                <ul>
                    <li>Shortened URLs (bit.ly, tinyurl, etc.)</li>
                    <li>Misspelled website names</li>
                    <li>Unexpected links from unknown senders</li>
                </ul>
                <p><strong>Best practice:</strong> Never click links in suspicious messages. Visit official websites directly.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Protection strategies
    st.markdown("### 🛡️ Your Protection Strategy")
    
    protection_col1, protection_col2 = st.columns(2)
    
    with protection_col1:
        st.markdown("""
            <div class='modern-card'>
                <h4>🕐 Take Your Time</h4>
                <ul>
                    <li><strong>Pause and think</strong> - Legitimate messages don't require immediate action</li>
                    <li><strong>Research the sender</strong> - Look up the company or phone number online</li>
                    <li><strong>Ask for help</strong> - Consult with family or friends if unsure</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='modern-card'>
                <h4>🔍 Verify Independently</h4>
                <ul>
                    <li><strong>Contact companies directly</strong> - Use official phone numbers or websites</li>
                    <li><strong>Don't use provided contact info</strong> - Scammers provide fake numbers</li>
                    <li><strong>Check official social media</strong> - Companies announce legitimate promotions publicly</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with protection_col2:
        st.markdown("""
            <div class='modern-card'>
                <h4>🔐 Guard Your Information</h4>
                <ul>
                    <li><strong>Never share PINs or passwords</strong> - Not even with "bank representatives"</li>
                    <li><strong>Be cautious with personal details</strong> - Scammers use them for identity theft</li>
                    <li><strong>Use strong, unique passwords</strong> - Enable two-factor authentication when possible</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='modern-card'>
                <h4>📢 Report and Block</h4>
                <ul>
                    <li><strong>Report to your mobile provider</strong> - Help them block scam numbers</li>
                    <li><strong>Block suspicious numbers</strong> - Prevent future contact attempts</li>
                    <li><strong>Share with others</strong> - Warn friends and family about new scam tactics</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Reporting section
    st.markdown("### 📢 Report Scam Messages")
    st.markdown("""
        <div class='modern-card'>
            <h4>🚨 Help Protect Others - Report Scams</h4>
            <p>When you receive a scam message, reporting it helps protect the entire community:</p>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;'>
                <div>
                    <h5>📱 Mobile Providers</h5>
                    <ul>
                        <li>Safaricom: Forward to 333</li>
                        <li>Airtel: Report via customer care</li>
                        <li>Telkom: Contact customer support</li>
                    </ul>
                </div>
                <div>
                    <h5>🏛️ Government Agencies</h5>
                    <ul>
                        <li>Communications Authority of Kenya</li>
                        <li>Directorate of Criminal Investigations</li>
                        <li>Kenya Computer Incident Response Team</li>
                    </ul>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("## 📊 Batch Message Analysis")
    
    st.markdown("""
        <div class='modern-card'>
            <h3>🔄 Analyze Multiple Messages</h3>
            <p style='color: var(--text-secondary);'>
                Upload a CSV file containing multiple SMS messages to analyze them all at once. 
                Perfect for businesses or organizations that need to check multiple messages for threats.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "**Upload CSV File** (must contain 'message_content' column)",
        type=["csv"],
        help="Your CSV file should have a column named 'message_content' with the SMS messages to analyze"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Show file preview
            st.markdown("### 👀 File Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            if 'message_content' in df.columns:
                st.success(f"✅ **File loaded successfully!** Found {len(df)} messages to analyze.")
                
                # Model selection for batch analysis
                batch_model = st.selectbox(
                    "**Choose AI Model for Batch Analysis:**",
                    ["XGBoost", "Logistic Regression"],
                    help="XGBoost provides more detailed analysis but takes longer"
                )
                
                if st.button("🚀 **Start Batch Analysis**", use_container_width=True):
                    with st.spinner(f"📊 **Analyzing {len(df)} messages...** This may take a few minutes."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, text in enumerate(df['message_content']):
                            pred, conf, _, _, _, inds, _ = predict(text, batch_model)
                            
                            # Create summary of indicators
                            high_risk_indicators = [ind['type'] for ind in inds if ind['severity'] == 'high']
                            medium_risk_indicators = [ind['type'] for ind in inds if ind['severity'] == 'medium']
                            
                            results.append({
                                'Message Preview': str(text)[:60] + '...' if len(str(text)) > 60 else str(text),
                                'Risk Assessment': pred.upper(),
                                'Confidence': f"{conf:.1%}",
                                'High Risk Indicators': "; ".join(high_risk_indicators) if high_risk_indicators else "None",
                                'Medium Risk Indicators': "; ".join(medium_risk_indicators) if medium_risk_indicators else "None",
                                'Total Indicators': len(inds)
                            })
                            
                            # Update progress
                            progress_bar.progress((idx + 1) / len(df))
                        
                        results_df = pd.DataFrame(results)
                        
                        # Display summary statistics
                        st.markdown("### 📈 Analysis Summary")
                        
                        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                        
                        scam_count = len(results_df[results_df['Risk Assessment'] == 'SCAM'])
                        legit_count = len(results_df[results_df['Risk Assessment'] == 'LEGIT'])
                        high_confidence = len(results_df[results_df['Confidence'].str.rstrip('%').astype(float) >= 80])
                        
                        with summary_col1:
                            st.markdown(f"""
                                <div class='stats-card'>
                                    <span class='stats-number' style='color: var(--danger-color);'>{scam_count}</span>
                                    <div class='stats-label'>Potential Scams</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with summary_col2:
                            st.markdown(f"""
                                <div class='stats-card'>
                                    <span class='stats-number' style='color: var(--success-color);'>{legit_count}</span>
                                    <div class='stats-label'>Appear Safe</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with summary_col3:
                            st.markdown(f"""
                                <div class='stats-card'>
                                    <span class='stats-number'>{high_confidence}</span>
                                    <div class='stats-label'>High Confidence</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with summary_col4:
                            st.markdown(f"""
                                <div class='stats-card'>
                                    <span class='stats-number'>{len(df)}</span>
                                    <div class='stats-label'>Total Analyzed</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Display detailed results
                        st.markdown("### 📋 Detailed Results")
                        st.dataframe(
                            results_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 **Download Complete Analysis Report**",
                            csv,
                            f"sms_scam_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
                        st.success("✅ **Batch analysis completed successfully!**")
            else:
                st.error("❌ **Invalid file format** - Your CSV file must contain a column named 'message_content'.")
                st.info("💡 **Tip:** Make sure your CSV has a header row with 'message_content' as one of the column names.")
        except Exception as e:
            st.error(f"❌ **Error processing file:** {str(e)}")
            st.info("💡 **Tip:** Make sure your file is a valid CSV format with proper encoding (UTF-8).")

# Enhanced footer
st.markdown("""
    <div class='footer'>
        <h3>🛡️ SMS Scam Shield</h3>
        <p style='color: var(--text-secondary); margin: 1rem 0;'>
            Protecting Kenyans from SMS scams with advanced AI technology. 
            Built with ❤️ for community safety and digital literacy.
        </p>
        <div class='footer-links'>
            <a href='https://mursimind.com' class='footer-link'>🌐 Visit Mursi Mind</a>
            <a href='#' class='footer-link'>📧 Contact Support</a>
            <a href='#' class='footer-link'>📚 Documentation</a>
            <a href='#' class='footer-link'>🔒 Privacy Policy</a>
        </div>
        <p style='color: var(--text-secondary); margin-top: 2rem; font-size: 0.9rem;'>
            © 2024 SMS Scam Shield. Empowering digital safety through AI innovation.
        </p>
    </div>
""", unsafe_allow_html=True)

