import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CMLPreprocessor:
    """Preprocessing pipeline for CML data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_features = [
            'commodity', 'material_type', 'feature_type', 'risk_level'
        ]
        self.numerical_features = [
            'design_thickness_mm', 'current_thickness_mm', 
            'average_corrosion_rate', 'remaining_life_years',
            'years_in_service', 'number_of_inspections', 'data_quality_score'
        ]
    
    def fit_transform(self, cmls: List[Any]) -> pd.DataFrame:
        """Fit preprocessor and transform CML data"""
        df = self._cmls_to_dataframe(cmls)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale numerical features
        numerical_cols = [c for c in self.numerical_features if c in df.columns]
        if numerical_cols:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        # Feature engineering
        df = self._engineer_features(df)
        
        return df
    
    def transform(self, cmls: List[Any]) -> pd.DataFrame:
        """Transform CML data using fitted preprocessor"""
        df = self._cmls_to_dataframe(cmls)
        df = self._handle_missing_values(df)
        
        # Encode categorical
        for col in self.categorical_features:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen labels
                df[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Scale numerical
        numerical_cols = [c for c in self.numerical_features if c in df.columns]
        if numerical_cols:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        df = self._engineer_features(df)
        
        return df
    
    def _cmls_to_dataframe(self, cmls: List[Any]) -> pd.DataFrame:
        """Convert CML objects to DataFrame"""
        data = []
        for cml in cmls:
            data.append({
                'cml_id': cml.cml_id,
                'commodity': cml.commodity,
                'material_type': cml.material_type,
                'feature_type': cml.feature_type,
                'risk_level': cml.risk_level.value if cml.risk_level else 'Unknown',
                'design_thickness_mm': cml.design_thickness_mm,
                'current_thickness_mm': cml.current_thickness_mm,
                'average_corrosion_rate': cml.average_corrosion_rate,
                'remaining_life_years': cml.remaining_life_years,
                'years_in_service': cml.years_in_service,
                'number_of_inspections': cml.number_of_inspections,
                'data_quality_score': cml.data_quality_score,
                'elimination_candidate': cml.elimination_candidate
            })
        return pd.DataFrame(data)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with imputation"""
        # Numerical: fill with median
        for col in self.numerical_features:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical: fill with mode
        for col in self.categorical_features:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        # Corrosion rate * years in service
        if 'average_corrosion_rate' in df.columns and 'years_in_service' in df.columns:
            df['total_corrosion_loss'] = df['average_corrosion_rate'] * df['years_in_service']
        
        # Thickness ratio
        if 'current_thickness_mm' in df.columns and 'design_thickness_mm' in df.columns:
            df['thickness_ratio'] = df['current_thickness_mm'] / (df['design_thickness_mm'] + 1e-6)
        
        # Inspection frequency
        if 'number_of_inspections' in df.columns and 'years_in_service' in df.columns:
            df['inspection_frequency'] = df['number_of_inspections'] / (df['years_in_service'] + 1)
        
        # Risk score (simple heuristic)
        if 'remaining_life_years' in df.columns:
            df['risk_score'] = 1 / (df['remaining_life_years'] + 1)
        
        return df
