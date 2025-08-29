# wordcloud_scam.py

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

def generate_wordcloud(csv_path: str, output_dir: str, text_column: str = "cleaned_text"):
    """
    Generate, display, and save a word cloud from a CSV column.

    Parameters:
    -----------
    csv_path : str
        Full path to the CSV file
    output_dir : str
        Full path to the output directory
    text_column : str
        Column name containing cleaned text
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the CSV.")

    # Combine all text entries
    text = " ".join(df[text_column].dropna().astype(str))

    # Generate word cloud
    wordcloud = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=200
    ).generate(text)

    # Plot
    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word cloud of common linguistic patterns in Kenyan SMS scams", fontsize=16)

    # Save output
    output_path = os.path.join(output_dir, "scam_wordcloud.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Word cloud saved to: {output_path}")


if __name__ == "__main__":
    csv_path = "/home/sjet/iwazolab/NLP-Scam-Detection/data/processed/scam_preprocessed.csv"
    output_dir = "/home/sjet/iwazolab/NLP-Scam-Detection/outputs/assets"
    generate_wordcloud(csv_path, output_dir, text_column="cleaned_text")
