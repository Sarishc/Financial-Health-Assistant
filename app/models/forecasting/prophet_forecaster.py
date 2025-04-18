# app/models/forecasting/prophet_forecaster.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from prophet import Prophet
import joblib
import os
import warnings

from app.models.forecasting.base_forecaster import BaseForecaster

class ProphetForecaster(BaseForecaster):
    """
    Prophet forecasting model implementation.
    Uses Facebook's Prophet model for time series forecasting.
    """
    
    def __init__(self, name: str = "prophet"):
        """
        Initialize the Prophet forecaster.
        
        Args:
            name: Name of the forecaster
        """
        super().__init__(name=name)
        self.model_params = {
            'daily_seasonality': 'auto',
            'weekly_seasonality': True,
            'yearly_seasonality': True,
            'seasonality_mode': 'multiplicative',
            'interval_width': 0.95,  # 95% confidence interval
            'changepoint_prior_scale': 0.05
        }
    
    def fit(self, time_series_data: Dict[str, pd.DataFrame], 
           value_col: str = 'amount_sum', 
           date_col: str = 'transaction_date',
           categories: Optional[List[str]] = None,
           **kwargs) -> None:
        """
        Fit Prophet models on the time series data.
        
        Args:
            time_series_data: Dictionary mapping category names to time series DataFrames
            value_col: Column name for the target values
            date_col: Column name for dates
            categories: List of categories to fit models for (if None, fit for all categories)
            **kwargs: Additional parameters for Prophet model
        """
        self.models = {}
        self.forecasts = {}
        
        # Update model parameters if provided
        if kwargs:
            for key, value in kwargs.items():
                if key in self.model_params:
                    self.model_params[key] = value
        
        # Process each time series
        for category, df in time_series_data.items():
            if categories is not None and category not in categories:
                continue
                
            # Prepare the time series
            ts_df = df.copy()
            
            # Prophet requires columns named 'ds' (date) and 'y' (value)
            if date_col in ts_df.columns:
                ts_df['ds'] = pd.to_datetime(ts_df[date_col])
            else:
                # If date is the index
                ts_df['ds'] = pd.to_datetime(ts_df.index)
            
            if value_col in ts_df.columns:
                ts_df['y'] = ts_df[value_col]
            else:
                print(f"Warning: Column '{value_col}' not found in DataFrame for category '{category}'")
                continue
            
            # Keep only required columns
            ts_df = ts_df[['ds', 'y']]
            
            # Remove missing values
            ts_df = ts_df.dropna(subset=['y'])
            
            # Check if we have enough data
            if len(ts_df) < 10:
                print(f"Warning: Not enough data for category '{category}' ({len(ts_df)} points)")
                continue
            
            try:
                # Initialize and fit Prophet model
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    
                    model = Prophet(
                        daily_seasonality=self.model_params['daily_seasonality'],
                        weekly_seasonality=self.model_params['weekly_seasonality'],
                        yearly_seasonality=self.model_params['yearly_seasonality'],
                        seasonality_mode=self.model_params['seasonality_mode'],
                        interval_width=self.model_params['interval_width'],
                        changepoint_prior_scale=self.model_params['changepoint_prior_scale']
                    )
                    
                    # Add additional regressors if provided in kwargs
                    for regressor in kwargs.get('add_regressors', []):
                        if regressor in ts_df.columns:
                            model.add_regressor(regressor)
                    
                    # Fit the model
                    model.fit(ts_df)
                
                # Store the fitted model
                self.models[category] = {
                    'model': model,
                    'last_date': ts_df['ds'].max(),
                    'parameters': self.model_params.copy()
                }
                
                print(f"Successfully fitted Prophet model for category '{category}'")
                
            except Exception as e:
                print(f"Error fitting Prophet model for category '{category}': {str(e)}")
        
        self.is_fitted = len(self.models) > 0
        
        if self.is_fitted:
            print(f"Successfully fitted {len(self.models)} Prophet models")
        else:
            print("Failed to fit any Prophet models")
    
    def predict(self, horizon: int = 30, 
               categories: Optional[List[str]] = None,
               freq: str = 'D',
               include_history: bool = False,
               **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for the specified horizon.
        
        Args:
            horizon: Number of future time periods to forecast
            categories: List of categories to generate forecasts for (if None, forecast for all categories)
            freq: Frequency of the forecast ('D' for daily, 'W' for weekly, etc.)
            include_history: Whether to include historical data in the forecast
            **kwargs: Additional parameters for forecast
            
        Returns:
            Dictionary mapping category names to DataFrames with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self.forecasts = {}
        
        if categories is None:
            categories = list(self.models.keys())
        
        for category in categories:
            if category not in self.models:
                print(f"Warning: No model found for category '{category}'")
                continue
                
            model_info = self.models[category]
            model = model_info['model']
            
            try:
                # Create future dataframe
                future = model.make_future_dataframe(periods=horizon, freq=freq, include_history=include_history)
                
                # Add additional regressors if provided
                for regressor in kwargs.get('add_regressors', []):
                    if 'regressor_data' in kwargs and regressor in kwargs['regressor_data']:
                        future[regressor] = kwargs['regressor_data'][regressor]
                
                # Generate forecast
                forecast = model.predict(future)
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'ds': forecast['ds'],
                    'amount': forecast['yhat'],
                    'lower_bound': forecast['yhat_lower'],
                    'upper_bound': forecast['yhat_upper']
                })
                
                # Store the forecast
                self.forecasts[category] = forecast_df
                
                print(f"Generated {horizon} step forecast for category '{category}'")
                
            except Exception as e:
                print(f"Error generating forecast for category '{category}': {str(e)}")
        
        return self.forecasts
    
    def save_model(self, output_dir: str = 'app/models/forecasting/saved_models') -> None:
        """
        Save trained Prophet models to disk.
        
        Args:
            output_dir: Directory to save the models
        """
        if not self.is_fitted:
            raise ValueError("No fitted models to save")
            
        os.makedirs(output_dir, exist_ok=True)
        
        for category, model_info in self.models.items():
            # Create safe filename
            safe_category = category.replace(' ', '_').lower()
            output_path = os.path.join(output_dir, f"{self.name}_{safe_category}_model.pkl")
            
            try:
                joblib.dump(model_info, output_path)
                print(f"Saved model for category '{category}' to {output_path}")
            except Exception as e:
                print(f"Error saving model for category '{category}': {str(e)}")
    
    def load_model(self, input_dir: str = 'app/models/forecasting/saved_models') -> None:
        """
        Load trained Prophet models from disk.
        
        Args:
            input_dir: Directory to load the models from
        """
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory '{input_dir}' does not exist")
            
        model_files = [f for f in os.listdir(input_dir) if f.startswith(f"{self.name}_") and f.endswith("_model.pkl")]
        
        if not model_files:
            raise ValueError(f"No {self.name} model files found in {input_dir}")
            
        self.models = {}
        
        for filename in model_files:
            input_path = os.path.join(input_dir, filename)
            
            try:
                # Extract category from filename
                category = filename.replace(f"{self.name}_", "").replace("_model.pkl", "").replace("_", " ")
                
                # Load the model
                model_info = joblib.load(input_path)
                self.models[category] = model_info
                
                print(f"Loaded model for category '{category}' from {input_path}")
            except Exception as e:
                print(f"Error loading model from {input_path}: {str(e)}")
        
        self.is_fitted = len(self.models) > 0
        
        if self.is_fitted:
            print(f"Successfully loaded {len(self.models)} Prophet models")
        else:
            print("Failed to load any Prophet models")