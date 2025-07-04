import os
import re
import logging
import pandas as pd
import numpy as np
import spacy
from langdetect import detect, DetectorFactory
import yaml

# ─────────────────────────────────────────────────────────────
# ✅ SETUP
# ─────────────────────────────────────────────────────────────

# Load config.yaml
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

input_path = config["preprocessing"]["input_path"]
output_path = config["preprocessing"]["output_path"]
sheng_keywords = config["preprocessing"]["sheng_keywords"]
phone_regex = config["preprocessing"]["phone_regex"]

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DetectorFactory.seed = 0
nlp = spacy.load("en_core_web_sm")

# ─────────────────────────────────────────────────────────────
# ✅ MAIN CLEANING FUNCTION
# ─────────────────────────────────────────────────────────────

def preprocess_text(text):
    """Preprocess text and extract features, handling invalid inputs."""
    # Handle NaN or non-string inputs
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
        tokens = ["empty"]

    # Extract features
    cap_words = len([word for word in original_text.split() if any(c.isupper() for c in word) and word.isalpha()])
    has_caps = cap_words > 0
    has_phone = bool(re.search(phone_regex, original_text))

    try:
        lang = detect(original_text)
        is_kiswahili_sheng = lang in ['sw', 'so']
    except:
        is_kiswahili_sheng = False

    has_sheng = any(keyword in original_text.lower() for keyword in sheng_keywords)
    is_kiswahili_sheng = is_kiswahili_sheng or has_sheng

    return tokens, cleaned_text, has_caps, has_phone, is_kiswahili_sheng

# ─────────────────────────────────────────────────────────────
# ✅ PIPELINE
# ─────────────────────────────────────────────────────────────

def run_preprocessing():
    logger.info(f"Reading dataset from {input_path}")
    df = pd.read_csv(input_path)
    if "message_content" not in df.columns:
        raise ValueError("'message_content' column is missing.")
    if "label" not in df.columns:
        raise ValueError("'label' column is missing.")

    logger.info("Applying preprocessing to all rows...")
    preprocessed_results = df["message_content"].apply(preprocess_text)
    results = list(zip(*preprocessed_results))
    df["tokens"], df["cleaned_text"], df["has_caps"], df["has_phone"], df["is_kiswahili_sheng"] = results

    logger.info(f"Label distribution before dropping NaN:\n{df['label'].value_counts(dropna=False)}")
    initial_rows = len(df)
    df = df.dropna(subset=["label"])
    logger.info(f"Dropped {initial_rows - len(df)} rows with NaN labels. Remaining: {len(df)}")

    # Evaluate
    logger.info(f"Label distribution:\n{df['label'].value_counts(dropna=False)}")
    logger.info(f"Messages with capitalized words: {df['has_caps'].sum()} ({df['has_caps'].mean()*100:.2f}%)")
    logger.info(f"Messages with phone numbers: {df['has_phone'].sum()} ({df['has_phone'].mean()*100:.2f}%)")
    logger.info(f"Messages with Kiswahili/Sheng: {df['is_kiswahili_sheng'].sum()} ({df['is_kiswahili_sheng'].mean()*100:.2f}%)")

    vocab = len(set(token for tokens in df["tokens"] for token in tokens))
    avg_tokens = df["tokens"].apply(len).mean()
    logger.info(f"Vocabulary size: {vocab}, Avg tokens per message: {avg_tokens:.2f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.drop(columns=["tokens"]).to_csv(output_path, index=False)
    logger.info(f"✅ Preprocessed data saved to {output_path}")

# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_preprocessing()