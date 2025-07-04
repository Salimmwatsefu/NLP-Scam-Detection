import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import pickle
import os
import logging
import yaml
from datetime import datetime

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise

def get_output_filename(base_dir, base_name, config):
    tag = config["run"]["tag"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M") if config["run"]["timestamp"] else ""
    filename = f"{base_name}_{tag}{'' if not timestamp else '_' + timestamp}"
    return os.path.join(base_dir, filename)

def extract_tfidf_features(df, config, vectorizer=None):
    logger.info("Extracting TF-IDF and binary features...")
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=config["features"]["max_features"],
            ngram_range=(1, 2),
            stop_words="english"
        )
        X_tfidf = vectorizer.fit_transform(df["cleaned_text"]).toarray()
    else:
        X_tfidf = vectorizer.transform(df["cleaned_text"]).toarray()
    
    binary_features = df[["has_caps", "has_phone", "is_kiswahili_sheng"]].astype(int).values
    X_combined = np.hstack([X_tfidf, binary_features])
    
    # Updated label_map for binary classification
    label_map = {"legit": 0, "scam": 1}
    y = df["label"].map(label_map).values
    if np.any(np.isnan(y)):
        logger.warning("NaN labels detected; dropping corresponding rows")
        valid_idx = ~np.isnan(y)
        X_combined = X_combined[valid_idx]
        y = y[valid_idx]
    
    return X_combined, y, vectorizer

def extract_bert_features(df, config, tokenizer=None):
    logger.info("Extracting BERT features...")
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    encodings = tokenizer(
        df["cleaned_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=config["features"]["bert_max_length"],
        return_tensors="pt"
    )
    # Updated label_map for binary classification
    label_map = {"legit": 0, "scam": 1}
    y = torch.tensor(df["label"].map(label_map).values)
    if torch.any(y.isnan()):
        logger.warning("NaN labels detected; dropping corresponding rows")
        valid_idx = ~y.isnan()
        encodings = {key: val[valid_idx] for key, val in encodings.items()}
        y = y[valid_idx]
    
    return encodings, y, tokenizer

def run_feature_extraction(config):
    input_path = config["preprocessing"]["output_path"]
    if not os.path.exists(input_path):
        logger.error(f"Preprocessed file {input_path} not found")
        raise FileNotFoundError(f"Preprocessed file {input_path} not found")
    
    logger.info(f"Reading preprocessed data from {input_path}")
    df = pd.read_csv(input_path)
    required_columns = ["cleaned_text", "label", "has_caps", "has_phone", "is_kiswahili_sheng"]
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Required columns {required_columns} not found")
        raise ValueError(f"Required columns {required_columns} not found")
    
    # Train/test split
    if config["model"]["test_split"] <= 0 or config["model"]["test_split"] >= 1:
        logger.error(f"Invalid test_split: {config['model']['test_split']}")
        raise ValueError("test_split must be between 0 and 1")
    
    train_df, test_df = train_test_split(
        df,
        test_size=config["model"]["test_split"],
        random_state=42,
        stratify=df["label"]
    )
    logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test samples")
    
    if train_df.empty or test_df.empty:
        logger.error("Train or test set is empty after split")
        raise ValueError("Train or test set is empty")

    result = {}
    # TF-IDF + binary features
    if config["features"]["method"] in ["tfidf", "all"]:
        X_train, y_train, vectorizer = extract_tfidf_features(train_df, config)
        X_test, y_test, _ = extract_tfidf_features(test_df, config, vectorizer=vectorizer)
        
        tfidf_train_path = get_output_filename(config["features"]["tfidf_features_dir"], "tfidf_features", config) + ".npy"
        tfidf_test_path = get_output_filename(config["features"]["tfidf_features_dir"], "test_tfidf_features", config) + ".npy"
        labels_train_path = get_output_filename(config["features"]["labels_dir"], "labels", config) + ".npy"
        labels_test_path = get_output_filename(config["features"]["labels_dir"], "test_labels", config) + ".npy"
        tfidf_vectorizer_path = get_output_filename(config["features"]["tfidf_vectorizer_dir"], "tfidf_vectorizer", config) + ".pkl"
        
        os.makedirs(config["features"]["tfidf_features_dir"], exist_ok=True)
        os.makedirs(config["features"]["labels_dir"], exist_ok=True)
        np.save(tfidf_train_path, X_train)
        np.save(tfidf_test_path, X_test)
        np.save(labels_train_path, y_train)
        np.save(labels_test_path, y_test)
        with open(tfidf_vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        logger.info(f"Saved: Train TF-IDF: {tfidf_train_path}, Test TF-IDF: {tfidf_test_path}")
        logger.info(f"Saved: Train labels: {labels_train_path}, Test labels: {labels_test_path}")
        logger.info(f"TF-IDF vectorizer: {tfidf_vectorizer_path}")
        
        result["tfidf"] = {
            "train": {"features": X_train, "labels": y_train},
            "test": {"features": X_test, "labels": y_test},
            "vectorizer": vectorizer
        }

    # BERT features
    if config["features"]["method"] in ["bert", "all"]:
        encodings_train, y_train_bert, tokenizer = extract_bert_features(train_df, config)
        encodings_test, y_test_bert, _ = extract_bert_features(test_df, config, tokenizer=tokenizer)
        
        bert_train_path = get_output_filename(config["features"]["bert_features_dir"], "bert_features", config) + ".pt"
        bert_test_path = get_output_filename(config["features"]["bert_features_dir"], "test_bert_features", config) + ".pt"
        os.makedirs(config["features"]["bert_features_dir"], exist_ok=True)
        torch.save(
            {
                "input_ids": encodings_train["input_ids"],
                "attention_mask": encodings_train["attention_mask"],
                "labels": y_train_bert
            },
            bert_train_path
        )
        torch.save(
            {
                "input_ids": encodings_test["input_ids"],
                "attention_mask": encodings_test["attention_mask"],
                "labels": y_test_bert
            },
            bert_test_path
        )
        logger.info(f"Saved: BERT train: {bert_train_path}, Test: {bert_test_path}")
        
        result["bert"] = {
            "train": {"encodings": encodings_train, "labels": y_train_bert},
            "test": {"encodings": encodings_test, "labels": y_test_bert},
            "tokenizer": tokenizer
        }

    logger.info("✅ Feature extraction completed.")
    return result

if __name__ == "__main__":
    config = load_config()
    run_feature_extraction(config)