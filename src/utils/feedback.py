import json
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
FEEDBACK_DIR = Path(__file__).parent.parent.parent / "data" / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "feedback_data.json"

def save_feedback(feedback_data):
    """
    Save user feedback to a JSON file.
    
    Args:
        feedback_data (dict): Dictionary containing feedback information
    """
    try:
        # Create feedback directory if it doesn't exist
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load existing feedback if file exists
        if FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, 'r') as f:
                feedback_history = json.load(f)
        else:
            feedback_history = []
        
        # Add new feedback
        feedback_history.append(feedback_data)
        
        # Save updated feedback
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(feedback_history, f, indent=2)
            
        logger.info(f"Feedback saved successfully: {feedback_data['message'][:50]}...")
        
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        raise

def load_feedback_data():
    """
    Load all feedback data from the JSON file.
    
    Returns:
        list: List of feedback data dictionaries
    """
    try:
        if FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading feedback data: {str(e)}")
        return []

def get_feedback_stats():
    """
    Get statistics about collected feedback.
    
    Returns:
        dict: Dictionary containing feedback statistics
    """
    feedback_data = load_feedback_data()
    
    stats = {
        "total_feedback": len(feedback_data),
        "feedback_by_label": {},
        "feedback_by_model": {},
        "recent_feedback": []
    }
    
    for feedback in feedback_data:
        # Count feedback by label
        label = feedback["correct_label"]
        stats["feedback_by_label"][label] = stats["feedback_by_label"].get(label, 0) + 1
        
        # Count feedback by model
        model = feedback.get("model_type", "unknown")
        stats["feedback_by_model"][model] = stats["feedback_by_model"].get(model, 0) + 1
        
        # Get recent feedback (last 5)
        if len(stats["recent_feedback"]) < 5:
            stats["recent_feedback"].append(feedback)
    
    return stats 