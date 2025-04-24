import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import nltk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import langid
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import gdown
import tarfile
import magic

# Initialize spaCy and NLTK
nlp = spacy.load("en_core_web_sm")
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    st.rerun()

# Define model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Download and extract models from Google Drive
@st.cache_resource
def download_models():
    try:
        # Google Drive file ID
        GOOGLE_DRIVE_FILE_ID = "1NxJCsmQesMVbDxmHfS8weuMGpFwxQGBW"
        GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        tar_path = os.path.join(MODEL_DIR, "models.tar.gz")
        
        # Download using gdown
        gdown.download(GOOGLE_DRIVE_URL, tar_path, quiet=False)
        
        # Verify file is a valid gzip
        file_type = magic.from_file(tar_path)
        if "gzip" not in file_type.lower():
            st.error(f"Downloaded file is not a gzip archive: {file_type}. Please check the Google Drive link.")
            raise ValueError(f"Invalid file type: {file_type}")
        
        # Extract
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=os.path.dirname(MODEL_DIR))
        
        # Clean up
        os.remove(tar_path)
    except Exception as e:
        st.error(f"Error downloading models: {str(e)}. Contact the app owner to verify the Google Drive link.")
        raise

# Load models and vectorizer
@st.cache_resource
def load_models():
    try:
        download_models()
        # BERT: Load tokenizer from bert-base-uncased, model from bert_scam_model
        bert_tokenizer_path = os.path.join(MODEL_DIR, 'bert-base-uncased')
        bert_model_path = os.path.join(MODEL_DIR, 'bert_scam_model')
        bert_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_path)
        bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)
        # Logistic Regression
        lr_model = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression_v1.joblib'))
        # XGBoost
        xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_v1.joblib'))
        # TF-IDF Vectorizer
        tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
        return bert_tokenizer, bert_model, lr_model, xgb_model, tfidf_vectorizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise

bert_tokenizer, bert_model, lr_model, xgb_model, tfidf_vectorizer = load_models()

# Your preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[\r\t\f\v]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'http\S+|www\S+|https\S+|[A-Za-z0-9.-]+\.(com|org|net)\b', '', text, flags=re.MULTILINE)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
    text = re.sub(r'\+\d{10,12}\b|\+\d{1,3}-\d{3}-\d{3}-\d{4}\b|\b(?:\+?254|0)(7\d{8}|11\d{7})\b', '', text)
    text = re.sub(r'\b[A-Z0-9]{10}\b|\bconfirmed\b|\bcompleted\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s]', ' ', text)

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha and len(token.text) > 2]
    cleaned_text = " ".join(tokens)
    return tokens, cleaned_text

# Extract additional features
def extract_features(text):
    tokens = word_tokenize(text)
    capitalized = sum(1 for word in tokens if word[0].isupper() and len(word) > 1)
    phone_pattern = r'\+\d{10,12}\b|\+\d{1,3}-\d{3}-\d{3}-\d{4}\b|\b(?:\+?254|0)(7\d{8}|11\d{7})\b'
    phone_numbers = len(re.findall(phone_pattern, text))
    lang, _ = langid.classify(text)
    is_kiswahili_sheng = lang in ['sw', 'mixed']
    return {
        'capitalized_count': capitalized,
        'phone_numbers': phone_numbers,
        'is_kiswahili_sheng': is_kiswahili_sheng
    }

# Prediction function
def predict(text, model_choice):
    tokens, cleaned_text = preprocess_text(text)
    features = extract_features(text)
    labels = ['Not Scam', 'Scam']

    if not cleaned_text.strip():
        return "Invalid Input", 0.0, tokens, cleaned_text, features

    if model_choice == "BERT":
        inputs = bert_tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][label_idx].item()
    else:
        tfidf_features = tfidf_vectorizer.transform([cleaned_text])
        model = lr_model if model_choice == "Logistic Regression" else xgb_model
        prob = model.predict_proba(tfidf_features)[0]
        label_idx = np.argmax(prob)
        confidence = prob[label_idx]

    return labels[label_idx], confidence, tokens, cleaned_text, features

# Streamlit app
st.title("Scam Detection App")
st.markdown("Select a model, enter text to check for scams, or upload a CSV file with a 'message_content' column.")

# Model selection
model_choice = st.selectbox("Choose Model", ["XGBoost", "BERT", "Logistic Regression"], index=0)

# Single text input
text_input = st.text_area("Enter text here:", height=150)
if st.button("Analyze Text"):
    if text_input:
        with st.spinner("Analyzing..."):
            prediction, confidence, tokens, cleaned_text, features = predict(text_input, model_choice)
            if prediction == "Invalid Input":
                st.warning("Input resulted in empty cleaned text. Please try a different message.")
            else:
                st.subheader("Results")
                st.write(f"**Model**: {model_choice}")
                st.write(f"**Prediction**: {prediction}")
                st.write(f"**Confidence**: {confidence:.2%}")
                
    else:
        st.warning("Please enter some text.")

# CSV file uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'message_content' in df.columns:
        st.subheader("Batch Analysis Results")
        results = []
        for text in df['message_content']:
            pred, conf, tokens, cleaned, feats = predict(text, model_choice)
            results.append({
                'Text': str(text)[:50] + '...' if len(str(text)) > 50 else str(text),
                'Prediction': pred,
                'Confidence': conf,
                'Tokens': str(tokens[:10]) + '...' if len(tokens) > 10 else str(tokens),
                'Capitalized': feats['capitalized_count'],
                'Phone Numbers': feats['phone_numbers'],
                'Kiswahili/Sheng': feats['is_kiswahili_sheng']
            })
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        st.download_button(
            label="Download Results",
            data=results_df.to_csv(index=False),
            file_name=f"scam_detection_results_{model_choice.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    else:
        st.error("CSV must contain a 'message_content' column.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, spaCy, Transformers, and scikit-learn for NLP Scam Detection")