import logging
from datetime import datetime
from config.loader import load_config
from data.preprocess import run_preprocessing
from features.extract import run_feature_extraction
from models.train import run_training

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline():
    config = load_config()
    
    logger.info("Starting NLP Scam Detection Pipeline...")
    
    try:
        # Step 1: Preprocessing
        logger.info("Running preprocessing...")
        run_preprocessing()
        
        # Step 2: Feature Extraction
        logger.info("Running feature extraction...")
        run_feature_extraction(config)
        
        # Step 3: Training
        logger.info("Running model training...")
        run_training(config)
        
        logger.info("✅ Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()