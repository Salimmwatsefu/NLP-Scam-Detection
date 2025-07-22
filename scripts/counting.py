import pandas as pd

# Load the CSV file
df = pd.read_csv('data/raw/scam_detection_labeled.csv')

# Count occurrences of each label
label_counts = df['label'].value_counts()

print(label_counts)
