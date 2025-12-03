import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import logging
from typing import List, Dict, Any
from app.ml.preprocess import CMLPreprocessor
from app.core.config import settings

logger = logging.getLogger(__name__)

class CMLEliminationModel:
    """XGBoost model for CML elimination prediction"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        self.preprocessor = CMLPreprocessor()
        self.feature_columns = None
        self.metrics = {}
        
        # Try to load existing model
        if os.path.exists(self.model_path):
            self.load_model()
    
    def train(self, cmls: List[Any], test_size: float = 0.2):
        """Train the elimination model"""
        logger.info(f"Training elimination model on {len(cmls)} CMLs...")
        
        # Preprocess data
        df = self.preprocessor.fit_transform(cmls)
        
        # Prepare features and target
        feature_cols = [c for c in df.columns if c not in ['cml_id', 'elimination_candidate']]
        X = df[feature_cols]
        y = df['elimination_candidate'].astype(int)
        
        self.feature_columns = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42,
            eval_metric='auc'
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Calculate metrics
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info(f"Model trained. Metrics: {self.metrics}")
        
        # Save model
        self.save_model()
    
    def predict(self, cmls: List[Any], threshold: float = 0.7) -> Dict[str, Dict]:
        """Predict elimination recommendations for CMLs"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Preprocess
        df = self.preprocessor.transform(cmls)
        X = df[self.feature_columns]
        
        # Predict
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        # Build results
        results = {}
        for i, cml in enumerate(cmls):
            results[cml.cml_id] = {
                'probability': float(probabilities[i]),
                'confidence': float(max(probabilities[i], 1 - probabilities[i])),
                'recommendation': 'eliminate' if predictions[i] == 1 else 'keep',
                'threshold_used': threshold
            }
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None or self.feature_columns is None:
            return {}
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_columns, importances.tolist()))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model metrics"""
        return self.metrics
    
    def save_model(self):
        """Save model to disk"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.feature_columns = model_data['feature_columns']
            self.metrics = model_data.get('metrics', {})
            
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
