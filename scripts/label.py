import pandas as pd

df = pd.read_csv('/home/sjet/iwazolab/NLP-Scam-Detection/data/prelabeled_messages.csv')

df['prelabel_numeric'] = df['prelabel'].str.split(':').str[0].astype(int)


df['label'] = df['prelabel_numeric'].map({
    2: 'scam',
    1: 'scam',
    0: 'legit'
})


df = df.drop('prelabel_numeric', axis=1)


df.to_csv('/home/sjet/iwazolab/NLP-Scam-Detection/data/scam_detection_labeled.csv', index=False)


print(df.head())