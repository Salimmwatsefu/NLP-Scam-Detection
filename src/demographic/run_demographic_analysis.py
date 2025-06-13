import pandas as pd
import os
from demographic_inference import infer_demographics
from demographic_training import train_demographic_models
from sklearn.preprocessing import LabelEncoder

# Load processed data
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                        "data", "processed", "scam_preprocessed.csv")
df = pd.read_csv(data_path)

# Get messages and labels
messages = df['message_content'].tolist()

# Remove object columns and label column
object_columns = df.select_dtypes(include=['object']).columns
X = df.drop(['message_content', 'label'] + list(object_columns), axis=1)  # Features

# Create label mapping
label_map = {
    'high_scam': 2,
    'legit': 0,
    'moderate_scam': 1
}

# Map labels directly
y = df['label'].map(label_map).values

print("\nLabel mapping:")
for label, idx in label_map.items():
    print(f"{idx}: {label}")

# Step 1: Infer demographics
print("\nInferring demographics...")
demographic_labels = infer_demographics(messages)
print(f"Demographic distribution:\n{demographic_labels.value_counts()}")

# Step 2: Train models with demographic features
print("\nTraining demographic-aware models...")
results = train_demographic_models(X, y, demographic_labels)

# Print metrics
print("\nModel Performance:")
for model_name, metrics in results['metrics'].items():
    print(f"\n{model_name.upper()}:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro Avg F1: {metrics['macro avg']['f1-score']:.3f}")
    print(f"Macro Avg Precision: {metrics['macro avg']['precision']:.3f}")
    print(f"Macro Avg Recall: {metrics['macro avg']['recall']:.3f}")
    
    # Print per-class metrics
    print("\nPer-class metrics:")
    for label, idx in label_map.items():
        print(f"\n{label}:")
        print(f"Precision: {metrics[str(idx)]['precision']:.3f}")
        print(f"Recall: {metrics[str(idx)]['recall']:.3f}")
        print(f"F1-score: {metrics[str(idx)]['f1-score']:.3f}")
        print(f"Support: {metrics[str(idx)]['support']}") 