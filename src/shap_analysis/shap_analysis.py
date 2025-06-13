import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Absolute file paths
DATA_PATH = '/home/sjet/iwazolab/NLP-Scam-Detection/data/processed/scam_preprocessed.csv'
TFIDF_VECTORIZER_PATH = '/home/sjet/iwazolab/NLP-Scam-Detection/data/features/tfidf/tfidf_vectorizer_scam_run_2025-05-25_01-47.pkl'
XGB_MODEL_PATH = '/home/sjet/iwazolab/NLP-Scam-Detection/outputs/models/xgboost_scam_run_2025-05-25_01-47.pkl'
LR_MODEL_PATH = '/home/sjet/iwazolab/NLP-Scam-Detection/outputs/models/logistic_scam_run_2025-05-25_01-47.pkl'

# Output paths
OUTPUT_DIR = '/home/sjet/iwazolab/NLP-Scam-Detection/outputs/shap_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class labels mapping
LABEL_MAPPING = {
    'legit': 0,
    'moderate_scam': 1,
    'high_scam': 2
}
CLASS_NAMES = ['Legit (Low-risk)', 'Moderate-risk scam', 'High-risk scam']

# Psychological trigger taxonomy
TRIGGER_TAXONOMY = {
    # Greed triggers
    'kes': 'greed',
    'win': 'greed',
    'congratulation': 'greed',
    'prize': 'greed',
    'free': 'greed',
    'million': 'greed',
    'bonus': 'greed',
    'reward': 'greed',
    'cash': 'greed',
    'money': 'greed',
    'payment': 'greed',
    'account': 'greed',
    'bank': 'greed',
    'loan': 'greed',
    'deposit': 'greed',
    'balance': 'greed',
    'fund': 'greed',
    'jackpot': 'greed',
    'bet': 'greed',
    'play': 'greed',
    'withdraw': 'greed',
    'refund': 'greed',
    'stake': 'greed',
    'earn': 'greed',
    'salary': 'greed',
    
    # Urgency triggers
    'usipitwe': 'urgency',
    'now': 'urgency',
    'immediately': 'urgency',
    'claim': 'urgency',
    'stop': 'urgency',
    'today': 'urgency',
    'time': 'urgency',
    'hurry': 'urgency',
    'quick': 'urgency',
    'last': 'urgency',
    'expire': 'urgency',
    'deadline': 'urgency',
    'soon': 'urgency',
    'instant': 'urgency',
    'rush': 'urgency',
    'limited': 'urgency',
    'offer': 'urgency',
    'flash': 'urgency',
    'sale': 'urgency',
    
    # Authority triggers
    'official': 'authority',
    'government': 'authority',
    'bank': 'authority',
    'security': 'authority',
    'verify': 'authority',
    'confirm': 'authority',
    'system': 'authority',
    'service': 'authority',
    'management': 'authority',
    'notice': 'authority',
    'secure': 'authority',
    'approved': 'authority',
    'registered': 'authority',
    'valid': 'authority',
    'comply': 'authority',
    
    # Fear triggers
    'suspend': 'fear',
    'block': 'fear',
    'security': 'fear',
    'fraud': 'fear',
    'warning': 'fear',
    'alert': 'fear',
    'unauthorized': 'fear',
    'suspicious': 'fear',
    'lock': 'fear',
    'loss': 'fear',
    'problem': 'fear',
    'issue': 'fear',
    'risk': 'fear',
    'danger': 'fear',
    
    # Social proof triggers
    'dear': 'social_proof',
    'friend': 'social_proof',
    'customer': 'social_proof',
    'member': 'social_proof',
    'valued': 'social_proof',
    'tenant': 'social_proof',
    'client': 'social_proof',
    'partner': 'social_proof',
    'user': 'social_proof',
    'subscriber': 'social_proof',
    
    # Language indicators
    'is_kiswahili_sheng': 'language',
    'has_caps': 'formatting',
    'has_phone': 'contact',
    
    # Scam-specific terms
    'mpesa': 'scam_specific',
    'paybill': 'scam_specific',
    'till': 'scam_specific',
    'promo': 'scam_specific',
    'code': 'scam_specific',
    'register': 'scam_specific',
    'signup': 'scam_specific',
    'link': 'scam_specific',
    'click': 'scam_specific',
    'visit': 'scam_specific',
    'dial': 'scam_specific',
    'sms': 'scam_specific',
    'whatsapp': 'scam_specific',
    'contact': 'scam_specific',
    'apply': 'scam_specific',
    'activate': 'scam_specific',
    'unlock': 'scam_specific',
    'redeem': 'scam_specific'
}

def load_data():
    """Load and preprocess SMS dataset."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Verify columns
    required_columns = ['message_content', 'label', 'has_caps', 'has_phone', 'is_kiswahili_sheng']
    if not all(col in df.columns for col in required_columns):
        raise KeyError(f"Dataset must contain {required_columns}")
    
    # Convert string labels to numerical
    df['label_numeric'] = df['label'].map(LABEL_MAPPING)
    if df['label_numeric'].isna().any():
        invalid_rows = df[df['label_numeric'].isna()][['label']]
        print(f"Warning: {len(invalid_rows)} rows with unmappable labels:\n{invalid_rows['label'].value_counts()}")
        df = df.dropna(subset=['label_numeric'])
        if df.empty:
            raise ValueError("No valid labels remain after dropping unmappable labels.")
    
    # Load pretrained vectorizer
    if not os.path.exists(TFIDF_VECTORIZER_PATH):
        raise FileNotFoundError(f"Pretrained vectorizer not found at {TFIDF_VECTORIZER_PATH}")
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    X_tfidf = vectorizer.transform(df['message_content'].fillna('')).toarray()
    
    # Combine TF-IDF features with binary features
    X_additional = df[['has_caps', 'has_phone', 'is_kiswahili_sheng']].astype(int).values
    X = np.hstack([X_tfidf, X_additional])
    y = df['label_numeric'].values
    
    # Combine feature names
    feature_names = np.concatenate([vectorizer.get_feature_names_out(), ['has_caps', 'has_phone', 'is_kiswahili_sheng']])
    
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Feature names length:", len(feature_names))
    
    return X, y, feature_names, df['message_content'].values

def analyze_model(model, model_type, X, feature_names, class_idx):
    """Analyze a single class using SHAP."""
    # Create explainer
    if model_type == 'xgboost':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X)
    
    # Get SHAP values
    shap_values = explainer.shap_values(X)
    
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        class_shap = shap_values[class_idx]
    else:
        # If we get a single array, it might be 3D (samples, features, classes)
        if shap_values.ndim == 3:
            class_shap = shap_values[:, :, class_idx]
        else:
            class_shap = shap_values
    
    # Ensure class_shap is 2D (samples, features)
    if class_shap.ndim != 2:
        raise ValueError(f"Unexpected SHAP values shape: {class_shap.shape}")
    
    # Create summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        class_shap,
        X,
        feature_names=feature_names,
        show=False,
        plot_size=(10, 6)
    )
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, f'{model_type}_class_{class_idx}_summary.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Calculate feature importance (mean absolute SHAP values)
    mean_shap = np.abs(class_shap).mean(axis=0)
    
    # Create DataFrame with feature names and their importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_value': mean_shap
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('shap_value', ascending=False)
    
    # Map triggers
    feature_importance['trigger'] = feature_importance['feature'].map(TRIGGER_TAXONOMY)
    
    # Save results
    result_path = os.path.join(OUTPUT_DIR, f'{model_type}_class_{class_idx}_importance.csv')
    feature_importance.head(20).to_csv(result_path, index=False)
    
    # Print summary of triggers
    trigger_summary = feature_importance.dropna(subset=['trigger']).groupby('trigger')['shap_value'].sum()
    print(f"\nTrigger Summary for {CLASS_NAMES[class_idx]}:")
    print(trigger_summary.sort_values(ascending=False))
    
    return feature_importance

def main():
    # Load data
    X, y, feature_names, messages = load_data()
    
    # Load models
    xgb_model = joblib.load(XGB_MODEL_PATH)
    lr_model = joblib.load(LR_MODEL_PATH)
    
    print("XGBoost model feature count:", xgb_model.n_features_in_)
    print("Logistic Regression model feature count:", lr_model.n_features_in_)
    
    # Analyze each class for both models
    for model_type, model in [('xgboost', xgb_model), ('logistic', lr_model)]:
        print(f"\nAnalyzing {model_type} model...")
        for class_idx in range(3):  # 0: legit, 1: moderate, 2: high
            print(f"Analyzing class {class_idx} ({CLASS_NAMES[class_idx]})...")
            feature_importance = analyze_model(model, model_type, X, feature_names, class_idx)
            print(f"Top 5 important features for {CLASS_NAMES[class_idx]}:")
            print(feature_importance.head())

if __name__ == "__main__":
    main()