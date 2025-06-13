"""
Script to run temporal analysis of scam patterns.
"""

import sys
import os
import pandas as pd
from pathlib import Path
import logging

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.temporal_analysis.temporal_analysis import (
    generate_synthetic_data,
    create_temporal_splits,
    analyze_pattern_evolution,
    retrain_model
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Set up paths
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "scam_preprocessed.csv"
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "temporal_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original data
    logger.info("Loading original data...")
    logger.info(f"Loading from: {data_path}")
    original_data = pd.read_csv(data_path)
    
    # Generate synthetic data
    logger.info("Generating synthetic data...")
    synthetic_data = generate_synthetic_data(original_data)
    synthetic_data.to_csv(output_dir / "synthetic_data.csv", index=False)
    
    # Create temporal splits
    logger.info("Creating temporal splits...")
    periods = create_temporal_splits(data_path, n_periods=3)
    
    # Analyze pattern evolution
    logger.info("Analyzing pattern evolution...")
    analysis_results = analyze_pattern_evolution(periods, output_dir)
    
    # Retrain models
    logger.info("Retraining models...")
    performance_metrics = retrain_model(periods, output_dir / "models")
    
    # Save results
    logger.info("Saving results...")
    pd.DataFrame(performance_metrics).to_csv(output_dir / "model_performance.csv")
    
    logger.info("Temporal analysis complete!")

if __name__ == "__main__":
    main() 