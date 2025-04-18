import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import joblib
import os
import warnings

from app.models.forecasting.base_forecaster import BaseForecaster

class ARIMAForecaster(BaseForecaster):
    """
    ARIMA forecasting model implementation.
    Uses ARIMA (AutoRegressive Integrated Moving Average) for time series forecasting.
    Uses statsmodels implementation instead of pmdarima.
    """
    
    def __init__(self, name: str = "arima", auto_order: bool = True):
        """
        Initialize the ARIMA forecaster.
        
        Args:
            name: Name of the forecaster
            auto_order: Whether to automatically determine ARIMA order
        """
        super().__init__(name=name)
        self.auto_order = auto_order
        self.model_params = {
            'default_order': (1, 1, 1),  # Default (p, d, q) values if auto_order is False
            'max_p': 5,
            'max_d': 2,
            'max_q': 5
        }
    
    def check_stationarity(self, series: pd.Series) -> Tuple[bool, float]:
        """
        Check if a time series is stationary using the Augmented Dickey-Fuller test.
        
        Args:
            series: Time series to test
            
        Returns:
            Tuple of (is_stationary, p_value)
        """
        # Remove NaN values
        series = series.dropna()
        
        if len(series) < 20:
            # Not enough data for reliable test, assume non-stationary
            return False, 1.0
            
        try:
            # Perform ADF test
            result = adfuller(series)
            p_value = result[1]
            
            # p-value less than 0.05 indicates stationarity
            is_stationary = p_value < 0.05
            
            return is_stationary, p_value
        except Exception as e:
            print(f"Error in stationarity check: {str(e)}")
            return False, 1.0
    
    def determine_arima_order(self, series: pd.Series) -> Tuple[int, int, int]:
        """
        Determine the best ARIMA order (p, d, q) using information criteria.
        This is a simplified version of auto_arima from pmdarima.
        
        Args:
            series: Time series data
            
        Returns:
            Tuple of (p, d, q) representing the best order
        """
        # Check stationarity to determine d
        is_stationary, _ = self.check_stationarity(series)
        
        if is_stationary:
            d = 0
        else:
            # Try first difference
            diff_series = series.diff().dropna()
            is_diff_stationary, _ = self.check_stationarity(diff_series)
            
            if is_diff_stationary:
                d = 1
            else:
                # Try second difference
                diff2_series = diff_series.diff().dropna()
                is_diff2_stationary, _ = self.check_stationarity(diff2_series)
                
                if is_diff2_stationary:
                    d = 2
                else:
                    # Default to d=1 if still not stationary
                    d = 1
        
        # Simple grid search for p and q
        best_aic = float('inf')
        best_order = (0, d, 0)
        
        max_p = min(self.model_params['max_p'], 3)  # Limit to 3 for faster execution
        max_q = min(self.model_params['max_q'], 3)  # Limit to 3 for faster execution
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    if p == 0 and q == 0:
                        continue  # Skip ARIMA(0,d,0)
                    
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        model_fit = model.fit()
                        aic = model_fit.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                    except Exception as e:
                        continue  # Skip orders that cause errors
        
        return best_order
    
    def fit(self, time_series_data: Dict[str, pd.DataFrame], 
           value_col: str = 'amount_sum', 
           date_col: str = 'transaction_date',
           categories: Optional[List[str]] = None,
           **kwargs) -> None:
        """
        Fit ARIMA models on the time series data.
        
        Args:
            time_series_data: Dictionary mapping category names to time series DataFrames
            value_col: Column name for the target values
            date_col: Column name for dates
            categories: List of categories to fit models for (if None, fit for all categories)
            **kwargs: Additional parameters for ARIMA model
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
            
            # Ensure date column is datetime and set as index
            if date_col in ts_df.columns:
                ts_df[date_col] = pd.to_datetime(ts_df[date_col])
                ts_df.set_index(date_col, inplace=True)
            
            # Sort by date
            ts_df = ts_df.sort_index()
            
            # Extract the target series
            if value_col in ts_df.columns:
                series = ts_df[value_col]
            else:
                print(f"Warning: Column '{value_col}' not found in DataFrame for category '{category}'")
                continue
            
            # Check if we have enough data
            if len(series) < 10:
                print(f"Warning: Not enough data for category '{category}' ({len(series)} points)")
                continue
            
            # Check stationarity
            is_stationary, p_value = self.check_stationarity(series)
            
            try:
                if self.auto_order:
                    # Determine the best order
                    best_order = self.determine_arima_order(series)
                    print(f"Category '{category}': Best ARIMA order = {best_order}")
                else:
                    # Use default parameters
                    best_order = self.model_params['default_order']
                
                # Fit the model
                model = ARIMA(series, order=best_order)
                model_fit = model.fit()
                
                # Store the fitted model
                self.models[category] = {
                    'model': model_fit,
                    'order': best_order,
                    'last_date': series.index[-1],
                    'freq': series.index.freq or pd.infer_freq(series.index) or 'D',
                    'is_stationary': is_stationary,
                    'p_value': p_value
                }
                
                print(f"Successfully fitted ARIMA model for category '{category}'")
                
            except Exception as e:
                print(f"Error fitting ARIMA model for category '{category}': {str(e)}")
        
        self.is_fitted = len(self.models) > 0
        
        if self.is_fitted:
            print(f"Successfully fitted {len(self.models)} ARIMA models")
        else:
            print("Failed to fit any ARIMA models")
    
    def predict(self, horizon: int = 30, 
               categories: Optional[List[str]] = None,
               return_conf_int: bool = True,
               alpha: float = 0.05,
               **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for the specified horizon.
        
        Args:
            horizon: Number of future time periods to forecast
            categories: List of categories to generate forecasts for (if None, forecast for all categories)
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals (default: 0.05 for 95% intervals)
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
            last_date = model_info['last_date']
            freq = model_info['freq']
            
            try:
                # Generate forecast
                forecast_result = model.get_forecast(steps=horizon)
                predicted_mean = forecast_result.predicted_mean
                
                if return_conf_int:
                    confidence_intervals = forecast_result.conf_int(alpha=alpha)
                    
                    # Extract lower and upper bounds
                    lower_bounds = confidence_intervals.iloc[:, 0]
                    upper_bounds = confidence_intervals.iloc[:, 1]
                
                # Generate date range for forecast
                forecast_index = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=horizon,
                    freq=freq
                )
                
                # Create forecast DataFrame
                if return_conf_int:
                    forecast_df = pd.DataFrame({
                        'ds': forecast_index,
                        'amount': predicted_mean.values,
                        'lower_bound': lower_bounds.values,
                        'upper_bound': upper_bounds.values
                    })
                else:
                    forecast_df = pd.DataFrame({
                        'ds': forecast_index,
                        'amount': predicted_mean.values
                    })
                
                # Store the forecast
                self.forecasts[category] = forecast_df
                
                print(f"Generated {horizon} step forecast for category '{category}'")
                
            except Exception as e:
                print(f"Error generating forecast for category '{category}': {str(e)}")
        
        return self.forecasts
    
    def save_model(self, output_dir: str = 'app/models/forecasting/saved_models') -> None:
        """
        Save trained ARIMA models to disk.
        
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
        Load trained ARIMA models from disk.
        
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
            print(f"Successfully loaded {len(self.models)} ARIMA models")
        else:
            print("Failed to load any ARIMA models")