import pandas as pd
import numpy as np
from typing import List, Dict

# Keywords and patterns for demographic inference
DEMOGRAPHIC_PATTERNS = {
    'youth': [
        'job', 'recruiter', 'work from home', 'online', 'internship',
        'career', 'opportunity', 'position', 'apply', 'resume',
        'salary', 'remote', 'work', 'employment', 'hire'
    ],
    'low_income': [
        'mpesa', 'till', 'paybill', 'deposit', 'withdraw',
        'cash', 'money', 'payment', 'loan', 'credit',
        'bank', 'account', 'balance', 'fund', 'transfer'
    ],
    'general': [
        'congratulation', 'winner', 'prize', 'award', 'bonus',
        'offer', 'discount', 'sale', 'free', 'gift',
        'claim', 'verify', 'confirm', 'update', 'security'
    ]
}

def infer_demographics(messages: List[str]) -> pd.Series:
    """
    Infer demographic target for each message based on keywords and patterns
    
    Args:
        messages: List of message texts
        
    Returns:
        Series of demographic labels (youth, low_income, general)
    """
    demographic_scores = {demo: [] for demo in DEMOGRAPHIC_PATTERNS.keys()}
    
    for message in messages:
        message = message.lower()
        scores = {}
        
        # Calculate score for each demographic
        for demo, patterns in DEMOGRAPHIC_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern in message)
            scores[demo] = score
        
        # Assign demographic based on highest score
        max_score = max(scores.values())
        if max_score > 0:
            # Get all demographics with the highest score
            top_demos = [demo for demo, score in scores.items() if score == max_score]
            # If multiple demographics have the same score, prefer youth > low_income > general
            if 'youth' in top_demos:
                demographic_scores['youth'].append(1)
                demographic_scores['low_income'].append(0)
                demographic_scores['general'].append(0)
            elif 'low_income' in top_demos:
                demographic_scores['youth'].append(0)
                demographic_scores['low_income'].append(1)
                demographic_scores['general'].append(0)
            else:
                demographic_scores['youth'].append(0)
                demographic_scores['low_income'].append(0)
                demographic_scores['general'].append(1)
        else:
            # If no patterns match, assign to general
            demographic_scores['youth'].append(0)
            demographic_scores['low_income'].append(0)
            demographic_scores['general'].append(1)
    
    # Convert scores to demographic labels
    demographics = []
    for i in range(len(messages)):
        if demographic_scores['youth'][i]:
            demographics.append('youth')
        elif demographic_scores['low_income'][i]:
            demographics.append('low_income')
        else:
            demographics.append('general')
    
    return pd.Series(demographics, name='demographic') 