# dataset_distribution.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_dataset_distribution(csv_path: str, output_dir: str, label_column: str = "label"):
    """
    Generate, display, and save a pie chart showing scam vs. non-scam distribution.

    Parameters:
    -----------
    csv_path : str
        Full path to the CSV file
    output_dir : str
        Full path to the output directory
    label_column : str
        Column name containing scam/non-scam labels
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path)

    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in the CSV.")

    # Count scam vs non-scam
    counts = df[label_column].value_counts()

    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=["#FF6F61", "#6BAED6"],  # scam=red, non-scam=blue
        textprops={"fontsize": 12}
    )
    plt.title("Distribution of SMS messages in the dataset (scam vs. legitimate)", fontsize=14)

    # Save
    output_path = os.path.join(output_dir, "dataset_distribution_pie.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Pie chart saved to: {output_path}")


if __name__ == "__main__":
    csv_path = "/home/sjet/iwazolab/NLP-Scam-Detection/data/processed/scam_preprocessed.csv"
    output_dir = "/home/sjet/iwazolab/NLP-Scam-Detection/outputs/assets"
    plot_dataset_distribution(csv_path, output_dir, label_column="label")
