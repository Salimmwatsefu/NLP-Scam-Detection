import pandas as pd

df = pd.read_csv('/home/sjet/iwazolab/NLP-Scam-Detection/data/prelabeled_messages.csv')

df['prelabel_numeric'] = df['prelabel'].str.split(':').str[0].astype(int)

# Create the new 'label' column based on the numeric prelabel
df['label'] = df['prelabel_numeric'].map({
    2: 'scam',
    1: 'scam',
    0: 'legit'
})

# Drop the temporary prelabel_numeric column (optional)
df = df.drop('prelabel_numeric', axis=1)

# Save the modified dataframe back to a CSV (optional)
df.to_csv('/home/sjet/iwazolab/NLP-Scam-Detection/data/scam_detection_labeled.csv', index=False)

# Display the first few rows to verify
print(df.head())