# linguistic_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import os

# Paths
data_path = "/home/sjet/iwazolab/NLP-Scam-Detection/data/processed/scam_preprocessed.csv"
output_dir = "/home/sjet/iwazolab/NLP-Scam-Detection/outputs/assets"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)

# Ensure necessary columns exist
if 'cleaned_text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must have 'cleaned_text' and 'label' columns")

# 1. Common words/phrases analysis
def get_top_words(text_series, n=20):
    all_words = " ".join(text_series).split()
    counter = Counter(all_words)
    return counter.most_common(n)

scam_words = get_top_words(df[df['label'] == 'scam']['cleaned_text'])
legit_words = get_top_words(df[df['label'] == 'legit']['cleaned_text'])

# Plot top words
def plot_top_words(word_counts, title, filename):
    words, counts = zip(*word_counts)
    plt.figure(figsize=(10,6))
    sns.barplot(x=list(counts), y=list(words), palette="viridis")
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

plot_top_words(scam_words, "Top 20 Words in Scam Messages", "top_scam_words.png")
plot_top_words(legit_words, "Top 20 Words in Legit Messages", "top_legit_words.png")

# 2. Message length distribution
df['msg_length'] = df['cleaned_text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10,6))
sns.histplot(df[df['label'] == 'scam']['msg_length'], color='red', label='Scam', kde=True, stat='density', bins=30)
sns.histplot(df[df['label'] == 'legit']['msg_length'], color='green', label='Legit', kde=True, stat='density', bins=30)
plt.title("Message Length Distribution (Words)")
plt.xlabel("Message Length (words)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "message_length_distribution.png"), dpi=300)
plt.close()

# 3. Urgency indicator frequency
urgency_keywords = ["urgent", "immediately", "now", "asap", "hurry", "action", "limited", "alert"]
def urgency_count(text):
    text = str(text).lower()
    return sum(word in text for word in urgency_keywords)

df['urgency_count'] = df['cleaned_text'].apply(urgency_count)

plt.figure(figsize=(8,6))
sns.boxplot(x='label', y='urgency_count', data=df, palette=['green','red'])
plt.title("Urgency Indicator Frequency by Label")
plt.ylabel("Urgency Word Count")
plt.xlabel("Label")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "urgency_indicator_frequency.png"), dpi=300)
plt.close()

print(f"✅ Linguistic pattern analysis charts saved in {output_dir}")
