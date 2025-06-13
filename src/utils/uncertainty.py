import numpy as np

def calculate_uncertainty(probability):
    """
    Calculate uncertainty score for a prediction probability.
    
    Args:
        probability (float): Prediction probability between 0 and 1
        
    Returns:
        float: Uncertainty score between 0 and 1
    """
    # Convert probability to uncertainty using entropy-like measure
    # Higher uncertainty when probability is closer to 0.5
    return 1 - 2 * abs(probability - 0.5)

def get_uncertainty_threshold():
    """
    Get the threshold for considering a prediction uncertain.
    
    Returns:
        float: Uncertainty threshold (default: 0.3)
    """
    return 0.3  # Can be adjusted based on model performance 