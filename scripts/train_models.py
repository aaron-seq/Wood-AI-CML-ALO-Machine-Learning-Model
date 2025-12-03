#!/usr/bin/env python
"""
Train ML models for CML Optimization

Usage:
    python scripts/train_models.py --data data/raw/CML_Optimization_Sample_Data.xlsx
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pandas as pd
from sqlalchemy.orm import Session
import logging

from backend.app.core.database import SessionLocal, engine, Base
from backend.app.models.db_models import CML, ModelTrainingRun
from backend.app.ml.model_elimination import CMLEliminationModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train CML Optimization ML models')
    parser.add_argument('--data', type=str, help='Path to training data Excel file')
    parser.add_argument('--output', type=str, default='data/models/', help='Output directory for models')
    args = parser.parse_args()
    
    logger.info("Starting model training...")
    
    # Create database session
    db: Session = SessionLocal()
    
    try:
        # Get CMLs from database or load from file
        if args.data and os.path.exists(args.data):
            logger.info(f"Loading data from {args.data}")
            # This would load data into database first
            # For now, assume data is already in database
        
        cmls = db.query(CML).all()
        
        if not cmls:
            logger.error("No CML data found. Please upload data first.")
            return
        
        logger.info(f"Found {len(cmls)} CMLs in database")
        
        # Train elimination model
        logger.info("Training CML Elimination Model...")
        model = CMLEliminationModel(model_path=os.path.join(args.output, 'cml_elimination_model.pkl'))
        model.train(cmls)
        
        # Save training run to database
        metrics = model.get_metrics()
        training_run = ModelTrainingRun(
            model_type='elimination',
            training_samples=metrics.get('train_samples', 0),
            validation_accuracy=metrics.get('accuracy', 0),
            test_accuracy=metrics.get('accuracy', 0),
            model_path=model.model_path,
            hyperparameters={
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
            },
            metrics=metrics,
            status='success'
        )
        db.add(training_run)
        db.commit()
        
        logger.info("✅ Model training completed successfully!")
        logger.info(f"Model metrics: {metrics}")
        logger.info(f"Model saved to: {model.model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()
