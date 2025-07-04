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
    page_title="SMS Scam Shield | Karibu",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Modern button styling */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1.5rem;
        border-radius: 0.75rem;
        background: linear-gradient(90deg, #7C3AED, #4F46E5);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
    }
    
    /* Risk level badges */
    .risk-badge {
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
    }
    .risk-moderate {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
    }
    .risk-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
    }
    
    /* Indicator cards */
    .indicator-card {
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
        transition: transform 0.2s ease;
    }
    .indicator-card:hover {
        transform: translateX(5px);
    }
    .indicator-high {
        border-left-color: #EF4444;
    }
    .indicator-medium {
        border-left-color: #F59E0B;
    }
    .indicator-low {
        border-left-color: #10B981;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700;
    }
    
    /* Protection tips */
    .protection-tip {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #7C3AED;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App Header with modern hero section
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 1rem;'>🛡️ SMS Scam Shield</h1>
        <p class='subtitle'>Protect yourself from SMS scams with advanced AI detection</p>
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

# Display model loading status
if lr_model is not None:
    st.success("Models loaded successfully!")
else:
    st.error("Failed to load models. Please ensure model files exist in the correct directories and try again.")

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
        explanation.append(f"The system is very confident ({confidence:.0%}) about this assessment.")
    elif confidence > 0.6:
        explanation.append(f"The system is moderately confident ({confidence:.0%}) about this assessment.")
    else:
        explanation.append(f"The system is less certain ({confidence:.0%}) about this assessment, please use extra caution.")
    
    # Summarize the key concerns
    high_severity_count = sum(1 for ind in indicators if ind['severity'] == 'high')
    medium_severity_count = sum(1 for ind in indicators if ind['severity'] == 'medium')
    
    if high_severity_count > 0 or medium_severity_count > 0:
        explanation.append("\n**Key Concerns Found:**")
        if high_severity_count > 0:
            explanation.append(f"- Found {high_severity_count} high-risk indicators 🔴")
        if medium_severity_count > 0:
            explanation.append(f"- Found {medium_severity_count} medium-risk indicators 🟡")
            
        # Add specific advice based on indicators
        for indicator in indicators:
            if indicator['severity'] in ['high', 'medium']:
                if indicator['type'] == 'Financial Bait':
                    explanation.append("- 💡 Be very careful with messages involving money or financial transactions")
                elif indicator['type'] == 'Suspicious Link':
                    explanation.append("- 💡 Never click on unexpected links, they might be phishing attempts")
                elif indicator['type'] == 'Personal Info Request':
                    explanation.append("- 💡 Legitimate services never ask for PINs or passwords via SMS")
                elif indicator['type'] == 'Betting Lure':
                    explanation.append("- 💡 Be skeptical of unexpected betting or gambling opportunities")
                elif indicator['type'] == 'Urgency Tactics':
                    explanation.append("- 💡 Scammers often create false urgency to pressure you")
    
    return "\n".join(explanation)

# Create modern tabs
tab1, tab2 = st.tabs(["🔍 Scan Message", "🎓 Safety Hub"])

with tab1:
    # Modern two-column layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Model selection with modern dropdown
        selected_model = st.selectbox(
            "Choose Analysis Engine:",
            ["Logistic Regression", "XGBoost"],
            help="Both engines are highly effective at detecting scams"
        )
        
        # Message input with modern textarea
        st.markdown("### Analyze Your Message")
        text_input = st.text_area(
            "Paste your SMS here:",
            height=100,
            help="Enter the message you want to check",
            placeholder="Type or paste the SMS message you want to analyze..."
        )
        
        # Modern analyze button
        if st.button("🔍 Analyze Message", use_container_width=True):
            if text_input:
                with st.spinner("🔄 Analyzing your message..."):
                    prediction, confidence, tokens, cleaned_text, features, indicators, _ = predict(text_input, selected_model)
                    
                    if prediction == "Invalid Input":
                        st.warning("⚠️ Please enter a valid message to analyze.")
                    elif prediction == "Error in prediction":
                        st.error("❌ Something went wrong. Please try again.")
                    else:
                        # Results section
                        st.markdown("### Analysis Results")
                        
                        # Risk level badge
                        risk_color = {
                            "scam": "risk-high",
                            "legit": "risk-low"
                        }[prediction]
                        
                        # Display results without any background containers
                        st.markdown(f"""
                            <div class='risk-badge {risk_color}'>
                                {prediction} • {confidence:.1%} Confidence
                            </div>
                            <div style='margin-top: 1rem;'>
                                {get_prediction_explanation(prediction, confidence, indicators)}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Indicators
                        if indicators:
                            st.markdown("#### 🔍 Detected Patterns")
                            for indicator in indicators:
                                severity_class = {
                                    'high': 'indicator-high',
                                    'medium': 'indicator-medium',
                                    'low': 'indicator-low'
                                }[indicator['severity']]
                                
                                st.markdown(f"""
                                    <div class='indicator-card {severity_class}'>
                                        <div style='display: flex; align-items: center; gap: 0.5rem;'>
                                            <span style='font-size: 1.2rem;'>{indicator['icon']}</span>
                                            <strong>{indicator['type']}</strong>
                                        </div>
                                        <p style='margin: 0.5rem 0 0 0;'>
                                            {indicator['detail']}
                                        </p>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        # Modern feedback section
                        with st.expander("📝 Help Us Improve"):
                            st.markdown("##### Was this analysis accurate?")
                            user_correction = st.radio(
                                "",
                                ["✅ Correct", "❌ Should be legit", "❌ Should be scam"]
                            )
                            
                            if st.button("Submit Feedback", use_container_width=True):
                                if user_correction != "✅ Correct":
                                    corrected_label = user_correction.split("Should be ")[-1]
                                    if save_feedback(text_input, prediction, corrected_label, selected_model):
                                        st.success("🙏 Thank you for helping us improve!")
                                    else:
                                        st.error("Failed to save feedback. Please try again.")
                                else:
                                    st.info("Thank you for confirming the prediction was correct!")
            else:
                st.warning("Please enter a message to analyze.")
        
        # Batch analysis in modern expandable section
        with st.expander("📊 Batch Analysis"):
            st.markdown("""
                <div style='background: white; padding: 1.5rem; border-radius: 1rem; margin: 1rem 0;'>
                    <h4>Analyze Multiple Messages</h4>
                    <p style='color: #6B7280;'>Upload a CSV file containing multiple messages to analyze them all at once.</p>
                </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Upload CSV (must contain 'message_content' column)", type=["csv"])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'message_content' in df.columns:
                        with st.spinner("📊 Processing messages..."):
                            results = []
                            for text in df['message_content']:
                                pred, conf, _, _, _, inds, _ = predict(text, selected_model)
                                results.append({
                                    'Message': str(text)[:50] + '...' if len(str(text)) > 50 else str(text),
                                    'Risk Level': pred,
                                    'Confidence': f"{conf:.1%}",
                                    'Indicators': "; ".join(ind['type'] for ind in inds)
                                })
                            results_df = pd.DataFrame(results)
                            
                            # Display results in a modern table
                            st.markdown("### Analysis Results")
                            st.dataframe(
                                results_df,
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Modern download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results",
                                csv,
                                "scam_detection_results.csv",
                                "text/csv",
                                use_container_width=True
                            )
                    else:
                        st.error("CSV must contain a 'message_content' column.")
                except Exception as e:
                    st.error(f"Error processing CSV: {str(e)}")

with tab2:
    st.markdown("## 🛡️ Your Guide to SMS Safety")
    
    # Common Scam Tactics
    st.markdown("### 🎯 Common Scam Tactics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            #### ⚡ Urgency Pressure
            Scammers create false urgency to make you act without thinking
        """)
    
    with col2:
        st.markdown("""
            #### 💰 Financial Lures
            Be cautious of unexpected money-related messages or too-good-to-be-true offers
        """)
    
    with col3:
        st.markdown("""
            #### 🔒 Personal Information
            Legitimate services never ask for PINs or passwords via SMS
        """)
    
    # Protection Tips
    st.markdown("### 🛡️ Protection Tips")
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
            #### 1. Take Your Time ⏳
            - Legitimate messages don't require immediate action
            - Think before you click or respond
            
            #### 2. Verify Independently ✅
            - Contact companies through their official channels
            - Don't use numbers or links from suspicious messages
        """)
    
    with tips_col2:
        st.markdown("""
            #### 3. Guard Your Information 🔐
            - Never share PINs, passwords, or personal details
            - Be cautious with any requests for sensitive data
            
            #### 4. Check URLs Carefully 🔍
            - Look for misspellings in web addresses
            - Verify the sender's identity before clicking links
        """)
    
    # Report Scams
    st.markdown("### 📢 Report Scams")
    st.markdown("""
        Help protect others by reporting scam messages to:
        - 📱 Your mobile service provider
        - 🏛️ Communications Authority of Kenya
        - 👮 Local cyber crime units
    """)

# Modern footer
st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #6B7280;'>
        <p>Built with ❤️ to protect Kenyans from SMS scams</p>
        <p>For support or feedback, visit <a href='https://mursimind.com' style='color: var(--primary-color); text-decoration: none;'>Mursi Mind</a></p>
    </div>
""", unsafe_allow_html=True)