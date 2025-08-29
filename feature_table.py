# feature_engineering_full.py

import pandas as pd
import os
import re

def extract_features(csv_path: str, output_dir: str, text_column: str = "cleaned_text"):
    """
    Extract actual features from preprocessed SMS messages and save a CSV table with values.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV.")

    # Scam keywords
    scam_keywords = ["urgent", "prize", "send", "church", "god bless", "mpesa", "congratulations", "code"]

    # Extract features
    features = pd.DataFrame()
    features['message'] = df[text_column]
    features['msg_length'] = df[text_column].apply(len)
    features['num_numbers'] = df[text_column].apply(lambda x: sum(c.isdigit() for c in str(x)))
    features['contains_greeting'] = df[text_column].apply(lambda x: any(word in str(x).lower() for word in ["hello", "dear", "hi"]))
    features['contains_scam_keyword'] = df[text_column].apply(lambda x: any(word in str(x).lower() for word in scam_keywords))
    features['num_exclamations'] = df[text_column].apply(lambda x: str(x).count('!'))
    features['num_uppercase_words'] = df[text_column].apply(lambda x: sum(1 for w in str(x).split() if w.isupper()))
    features['code_switching'] = df[text_column].apply(lambda x: bool(re.search(r'\b(ksh|mpesa|asap)\b', str(x).lower())))

    # Save table with actual feature values
    output_path = os.path.join(output_dir, "feature_engineering_full.csv")
    features.to_csv(output_path, index=False)
    print(f"✅ Feature engineering table saved to: {output_path}")

    return features

if __name__ == "__main__":
    csv_path = "/home/sjet/iwazolab/NLP-Scam-Detection/data/processed/scam_preprocessed.csv"
    output_dir = "/home/sjet/iwazolab/NLP-Scam-Detection/outputs/assets"
    features = extract_features(csv_path, output_dir, text_column="cleaned_text")
    print(features.head(10))
