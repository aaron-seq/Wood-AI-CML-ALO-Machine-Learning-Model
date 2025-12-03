import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ModelExplainer:
    """SHAP-based model explainability"""
    
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.explainer = None
    
    def initialize(self, background_data: pd.DataFrame):
        """Initialize SHAP explainer with background data"""
        try:
            # Use TreeExplainer for XGBoost
            self.explainer = shap.TreeExplainer(self.model, background_data)
            logger.info("SHAP explainer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
    
    def explain_prediction(self, X: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """
        Generate SHAP explanations for predictions
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
        
        Returns:
            Dictionary with SHAP values and explanations
        """
        if self.explainer is None:
            logger.warning("SHAP explainer not initialized")
            return {}
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # If binary classification, take positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Get base value
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1]
            
            explanations = []
            for i in range(len(X)):
                # Get top contributing features
                feature_contributions = list(zip(feature_names, shap_values[i]))
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                
                # Create human-readable explanation
                top_features = feature_contributions[:5]
                explanation_text = self._create_explanation(top_features)
                
                explanations.append({
                    'shap_values': dict(zip(feature_names, shap_values[i].tolist())),
                    'top_features': [
                        {'feature': f, 'impact': float(v)} 
                        for f, v in top_features
                    ],
                    'explanation': explanation_text,
                    'base_value': float(base_value)
                })
            
            return {
                'explanations': explanations,
                'feature_importance': self._get_global_importance(shap_values, feature_names)
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {}
    
    def _create_explanation(self, top_features: List[tuple]) -> str:
        """Create human-readable explanation from SHAP values"""
        positive_factors = []
        negative_factors = []
        
        for feature, value in top_features:
            if value > 0:
                positive_factors.append(feature)
            elif value < 0:
                negative_factors.append(feature)
        
        explanation_parts = []
        
        if positive_factors:
            explanation_parts.append(f"Factors increasing elimination probability: {', '.join(positive_factors)}")
        
        if negative_factors:
            explanation_parts.append(f"Factors decreasing elimination probability: {', '.join(negative_factors)}")
        
        return ". ".join(explanation_parts) if explanation_parts else "No significant factors identified"
    
    def _get_global_importance(self, shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate global feature importance from SHAP values"""
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        return dict(zip(feature_names, importance.tolist()))
