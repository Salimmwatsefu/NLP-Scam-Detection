"""
Temporal Analysis of Scam Tactics

This module implements temporal analysis of scam patterns, including:
- Synthetic data generation
- Temporal splitting
- Pattern evolution analysis
- Model adaptation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.models.train import train_model
from src.utils.preprocess import preprocess_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
REPLACEMENT_PATTERNS = {
    'KES': ['Bitcoin', 'Ethereum', 'USDT', 'crypto'],
    'M-Pesa': ['PayPal', 'Venmo', 'Cash App', 'mobile money'],
    'bank': ['wallet', 'account', 'crypto wallet'],
    'money': ['crypto', 'tokens', 'coins'],
    'deposit': ['transfer', 'send', 'invest'],
    'withdraw': ['cash out', 'convert', 'sell']
}

SENG_TERMS = [
    'mziki', 'pesa', 'chuma', 'dough', 'bread',
    'mula', 'stash', 'bag', 'stack', 'paper'
]

def generate_synthetic_data(
    original_data: pd.DataFrame,
    variation_ratio: float = 0.5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic variations of existing scam messages.
    
    Args:
        original_data: DataFrame containing original messages
        variation_ratio: Proportion of messages to generate variations for
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing synthetic messages
    """
    np.random.seed(seed)
    
    # Select messages to vary
    n_variations = int(len(original_data) * variation_ratio)
    messages_to_vary = original_data.sample(n=n_variations, random_state=seed)
    
    synthetic_messages = []
    for _, row in messages_to_vary.iterrows():
        # Generate variations
        message = row['message_content']
        
        # Replace patterns
        for old, new_options in REPLACEMENT_PATTERNS.items():
            if old.lower() in message.lower():
                new = np.random.choice(new_options)
                message = message.replace(old, new)
        
        # Add Sheng terms
        n_terms = np.random.randint(1, 3)
        terms = np.random.choice(SENG_TERMS, n_terms)
        words = message.split()
        for term in terms:
            pos = np.random.randint(0, len(words))
            words.insert(pos, term)
        message = ' '.join(words)
        
        synthetic_messages.append({
            'message_content': message,
            'label': row['label'],
            'is_synthetic': True,
            'original_message': row['message_content']
        })
    
    return pd.DataFrame(synthetic_messages)

def create_temporal_splits(data_path, n_periods=3):
    """Create temporal splits of the data"""
    logger.info(f"Creating {n_periods} temporal splits...")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Sort by date if available, otherwise use index
    if 'date' in df.columns:
        df = df.sort_values('date')
    else:
        df = df.reset_index()
    
    # Calculate period size
    period_size = len(df) // n_periods
    
    # Create periods
    periods = {}
    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = start_idx + period_size if i < n_periods - 1 else len(df)
        
        period_data = df.iloc[start_idx:end_idx].copy()
        periods[f'period_{i+1}'] = period_data
    
    return periods

def analyze_pattern_evolution(periods, output_dir):
    """Analyze how scam patterns evolve over time"""
    logger.info("Analyzing pattern evolution...")
    
    # Initialize results storage
    pattern_evolution = {
        'scam_indicators': [],
        'linguistic_shifts': [],
        'temporal_patterns': []
    }
    
    # Analyze each period
    for period_name, period_data in periods.items():
        logger.info(f"Analyzing period: {period_name}")
        
        # Extract scam indicators
        indicators = extract_scam_indicators(period_data['message_content'].tolist())
        pattern_evolution['scam_indicators'].append({
            'period': period_name,
            'indicators': indicators
        })
        
        # Analyze linguistic patterns
        linguistic_features = analyze_linguistic_patterns(period_data['message_content'].tolist())
        pattern_evolution['linguistic_shifts'].append({
            'period': period_name,
            'features': linguistic_features
        })
        
        # Track temporal patterns
        temporal_patterns = analyze_temporal_patterns(period_data)
        pattern_evolution['temporal_patterns'].append({
            'period': period_name,
            'patterns': temporal_patterns
        })
    
    # Generate visualizations
    generate_evolution_plots(pattern_evolution, output_dir)
    
    return pattern_evolution

def extract_scam_indicators(messages):
    """Extract common scam indicators from messages"""
    indicators = {
        'urgency': 0,
        'prize': 0,
        'account_related': 0,
        'money_related': 0,
        'free_offer': 0,
        'click_bait': 0
    }
    
    for message in messages:
        message = message.lower()
        if 'urgent' in message:
            indicators['urgency'] += 1
        if 'winner' in message:
            indicators['prize'] += 1
        if 'account' in message:
            indicators['account_related'] += 1
        if 'money' in message:
            indicators['money_related'] += 1
        if 'free' in message:
            indicators['free_offer'] += 1
        if 'click' in message:
            indicators['click_bait'] += 1
    
    # Convert to frequencies
    total = len(messages)
    return {k: v/total for k, v in indicators.items()}

def analyze_linguistic_patterns(messages):
    """Analyze linguistic patterns in messages"""
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(messages)
    feature_names = vectorizer.get_feature_names_out()
    mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    
    return {
        'features': list(feature_names),
        'scores': list(mean_scores)
    }

def analyze_temporal_patterns(data):
    """Analyze temporal patterns in the data"""
    patterns = {
        'scam_ratio': len(data[data['label'] != 'legit']) / len(data),
        'avg_message_length': data['message_content'].str.len().mean(),
        'unique_words': data['message_content'].str.split().apply(set).apply(len).mean()
    }
    return patterns

def generate_evolution_plots(pattern_evolution, output_dir):
    """Generate visualization plots for pattern evolution"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot scam indicators
    plt.figure(figsize=(12, 6))
    indicators_df = pd.DataFrame([
        {**period['indicators'], 'period': period['period']}
        for period in pattern_evolution['scam_indicators']
    ])
    
    for indicator in indicators_df.columns:
        if indicator != 'period':
            plt.plot(
                indicators_df['period'],
                indicators_df[indicator],
                label=indicator,
                marker='o'
            )
    
    plt.title('Evolution of Scam Indicators Over Time')
    plt.xlabel('Period')
    plt.ylabel('Indicator Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'scam_indicators.png')
    plt.close()
    
    # Plot linguistic shifts
    plt.figure(figsize=(12, 6))
    linguistic_data = []
    for period in pattern_evolution['linguistic_shifts']:
        for feature, score in zip(period['features']['features'], period['features']['scores']):
            linguistic_data.append({
                'period': period['period'],
                'feature': feature,
                'score': score
            })
    
    linguistic_df = pd.DataFrame(linguistic_data)
    pivot_df = linguistic_df.pivot(
        index='period',
        columns='feature',
        values='score'
    )
    sns.heatmap(pivot_df, cmap='YlOrRd')
    plt.title('Linguistic Shifts Over Time')
    plt.xlabel('Features')
    plt.ylabel('Period')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'linguistic_shifts.png')
    plt.close()

def retrain_model(periods, output_dir):
    """Retrain model on each time period"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create models directory if it doesn't exist
    models_dir = output_dir / "models" / "temporal_analysis"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    performance_metrics = {}
    for period_name, period_data in periods.items():
        logger.info(f"Retraining model for period: {period_name}")
        
        # Prepare data
        X_text = period_data['message_content'].apply(preprocess_text).values
        y = period_data['label'].values
        
        # Vectorize text data
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(X_text)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Train model
        model = train_model(
            X=X,
            y=y_encoded,
            model_type="XGBoost"
        )
        
        # Save model and vectorizer
        model_path = models_dir / f"model_{period_name}.pkl"
        vectorizer_path = models_dir / f"vectorizer_{period_name}.pkl"
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        logger.info(f"Saved model and vectorizer for {period_name} to {models_dir}")
        
        # Evaluate
        y_pred = model.predict(X)
        metrics = {
            'accuracy': accuracy_score(y_encoded, y_pred),
            'f1': f1_score(y_encoded, y_pred, average='weighted')
        }
        performance_metrics[period_name] = metrics
        
    return performance_metrics 