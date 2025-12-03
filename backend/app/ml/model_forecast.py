import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Literal
import logging

logger = logging.getLogger(__name__)

class CMLForecastModel:
    """Time-series forecasting for CML thickness"""
    
    def __init__(self, model_type: Literal['prophet', 'linear', 'arima'] = 'prophet'):
        self.model_type = model_type
        self.model = None
    
    def predict(self, historical_data: pd.DataFrame, periods: int = 24) -> pd.DataFrame:
        """
        Forecast future thickness
        
        Args:
            historical_data: DataFrame with 'ds' (date) and 'y' (thickness) columns
            periods: Number of months to forecast
        
        Returns:
            DataFrame with forecast including 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
        """
        if self.model_type == 'prophet':
            return self._prophet_forecast(historical_data, periods)
        elif self.model_type == 'linear':
            return self._linear_forecast(historical_data, periods)
        else:
            return self._linear_forecast(historical_data, periods)
    
    def _prophet_forecast(self, df: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Forecast using Facebook Prophet"""
        try:
            from prophet import Prophet
            
            # Create and fit model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            
            model.fit(df)
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=periods, freq='M')
            forecast = model.predict(future)
            
            # Return only future predictions
            forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast = forecast[forecast['ds'] > df['ds'].max()]
            
            return forecast
            
        except Exception as e:
            logger.warning(f"Prophet forecast failed: {e}. Falling back to linear model.")
            return self._linear_forecast(df, periods)
    
    def _linear_forecast(self, df: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Simple linear regression forecast"""
        from sklearn.linear_model import LinearRegression
        
        # Convert dates to numeric (days since first measurement)
        df = df.copy()
        df['days'] = (df['ds'] - df['ds'].min()).dt.days
        
        # Fit linear model
        X = df[['days']].values
        y = df['y'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate future dates
        last_date = df['ds'].max()
        future_dates = [last_date + timedelta(days=30*i) for i in range(1, periods+1)]
        future_days = np.array([(d - df['ds'].min()).days for d in future_dates]).reshape(-1, 1)
        
        # Predict
        predictions = model.predict(future_days)
        
        # Calculate prediction interval (simple estimate)
        residuals = y - model.predict(X)
        std_error = np.std(residuals)
        margin = 1.96 * std_error  # 95% confidence
        
        forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': predictions,
            'yhat_lower': predictions - margin,
            'yhat_upper': predictions + margin
        })
        
        return forecast
