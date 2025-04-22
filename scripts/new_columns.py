import re
import pandas as pd
import spacy
import nltk
from nltk.corpus import words

# Download NLTK words corpus (run once)
nltk.download('words', quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load English words set
english_words = set(words.words())

# Load dataset
df = pd.read_csv("/home/sjet/iwazolab/NLP-Scam-Detection/data/scam_detection_labeled.csv")

# 1. Capitalized Words Column
def has_capitalized_words(text):
    # Match words with 2+ letters, all uppercase
    pattern = r'\b[A-Z]{2,}\b'
    return 1 if re.search(pattern, str(text)) else 0

df["has_capitalized_words"] = df["message_content"].apply(has_capitalized_words)

# 2. Phone Numbers Column
def extract_phone_numbers(text):
    # Match full Kenyan/international phone numbers
    pattern = r'(?:\+\d{10,12}|\+\d{1,3}-\d{3}-\d{3}-\d{4}|(?:\+?254|0)(7\d{8}|11\d{7}))\b'
    matches = re.findall(pattern, str(text))
    return ", ".join(matches) if matches else ""

df["phone_numbers"] = df["message_content"].apply(extract_phone_numbers)

# 3. Kiswahili/Sheng Language Column
def has_kiswahili_sheng(text):
    # List of common Kiswahili/Sheng words
    kiswahili_sheng = [
        "usipitwe", "hii", "ingia", "pepea", "yako", "leo", "hapa", "ucheze",
        "upate", "kama", "mzee", "poa", "sawa", "shika", "mambo"
    ]
    text = str(text).lower()
    # Check if any listed word appears
    if any(word in text for word in kiswahili_sheng):
        return 1
    # Tokenize and check for non-English, non-name words
    doc = nlp(text)
    for token in doc:
        word = token.text
        if (token.is_alpha and len(word) > 2 and not token.is_stop and
            word not in english_words and word not in kiswahili_sheng):
            # Check if word is a name
            is_name = any(ent.text.lower() == word and ent.label_ == "PERSON" for ent in doc.ents)
            if not is_name:
                return 1
    return 0

df["has_kiswahili_sheng"] = df["message_content"].apply(has_kiswahili_sheng)

# Save updated DataFrame to same CSV
df.to_csv("/home/sjet/iwazolab/NLP-Scam-Detection/data/scam_detection_labeled.csv", index=False)
print("Updated dataset saved to '../data/scam_detection_labeled.csv'")