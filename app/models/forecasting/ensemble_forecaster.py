import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import joblib
import os

from app.models.forecasting.base_forecaster import BaseForecaster
from app.models.forecasting.arima_forecaster import ARIMAForecaster
from app.models.forecasting.prophet_forecaster import ProphetForecaster
# Removed LSTM import to avoid dependency issues

class EnsembleForecaster(BaseForecaster):
    """
    Ensemble forecasting model implementation.
    Combines predictions from multiple forecasting models.
    """
    
    def __init__(self, name: str = "ensemble"):
        """
        Initialize the Ensemble forecaster.
        
        Args:
            name: Name of the forecaster
        """
        super().__init__(name=name)
        self.forecasters = {}
        self.weights = {}
        self.model_params = {
            'combination_method': 'weighted_average',  # 'weighted_average', 'simple_average', 'median'
            'default_weights': {
                'arima': 0.4,
                'prophet': 0.6
                # Removed LSTM weight
            }
        }
    
    def add_forecaster(self, forecaster: BaseForecaster, weight: float = None) -> None:
        """
        Add a forecaster to the ensemble.
        
        Args:
            forecaster: Forecaster object to add
            weight: Weight to assign to this forecaster (if None, use default weight)
        """
        if not isinstance(forecaster, BaseForecaster):
            raise TypeError("forecaster must be an instance of BaseForecaster")
            
        self.forecasters[forecaster.name] = forecaster
        
        if weight is not None:
            self.weights[forecaster.name] = weight
        elif forecaster.name in self.model_params['default_weights']:
            self.weights[forecaster.name] = self.model_params['default_weights'][forecaster.name]
        else:
            # Assign equal weights if not specified
            self.weights[forecaster.name] = 1.0 / (len(self.forecasters) + 1)
            
        # Normalize weights
        total_weight = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total_weight
    
    def fit(self, time_series_data: Dict[str, pd.DataFrame], 
           value_col: str = 'amount_sum', 
           date_col: str = 'transaction_date',
           categories: Optional[List[str]] = None,
           **kwargs) -> None:
        """
        Fit all forecasters in the ensemble on the time series data.
        
        Args:
            time_series_data: Dictionary mapping category names to time series DataFrames
            value_col: Column name for the target values
            date_col: Column name for dates
            categories: List of categories to fit models for (if None, fit for all categories)
            **kwargs: Additional parameters for forecasters
        """
        if not self.forecasters:
            raise ValueError("No forecasters in the ensemble. Use add_forecaster() to add forecasters.")
            
        self.models = {}
        self.forecasts = {}
        
        # Update model parameters if provided
        if kwargs:
            for key, value in kwargs.items():
                if key in self.model_params:
                    self.model_params[key] = value
        
        # Fit each forecaster
        for name, forecaster in self.forecasters.items():
            print(f"Fitting {name} forecaster...")
            
            try:
                # Extract forecaster-specific kwargs if present
                forecaster_kwargs = kwargs.get(name, {})
                
                # Fit the forecaster
                forecaster.fit(
                    time_series_data,
                    value_col=value_col,
                    date_col=date_col,
                    categories=categories,
                    **forecaster_kwargs
                )
                
                print(f"Successfully fitted {name} forecaster")
            except Exception as e:
                print(f"Error fitting {name} forecaster: {str(e)}")
        
        # Store fitted forecasters
        for name, forecaster in self.forecasters.items():
            if forecaster.is_fitted:
                self.models[name] = forecaster.models
        
        self.is_fitted = len(self.models) > 0
        
        if self.is_fitted:
            print(f"Successfully fitted {len(self.models)} ensemble forecasters")
        else:
            print("Failed to fit any ensemble forecasters")
    
    def predict(self, horizon: int = 30, 
               categories: Optional[List[str]] = None,
               combination_method: Optional[str] = None,
               **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Generate ensemble forecasts for the specified horizon.
        
        Args:
            horizon: Number of future time periods to forecast
            categories: List of categories to generate forecasts for (if None, forecast for all categories)
            combination_method: Method to combine forecasts ('weighted_average', 'simple_average', 'median')
            **kwargs: Additional parameters for forecast
            
        Returns:
            Dictionary mapping category names to DataFrames with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self.forecasts = {}
        
        # Use specified combination method or default
        if combination_method is None:
            combination_method = self.model_params['combination_method']
        
        # Get forecasts from each forecaster
        forecaster_predictions = {}
        
        for name, forecaster in self.forecasters.items():
            if not forecaster.is_fitted:
                print(f"Warning: {name} forecaster is not fitted")
                continue
                
            # Generate forecasts
            try:
                # Extract forecaster-specific kwargs if present
                forecaster_kwargs = kwargs.get(name, {})
                
                predictions = forecaster.predict(
                    horizon=horizon,
                    categories=categories,
                    **forecaster_kwargs
                )
                
                forecaster_predictions[name] = predictions
                print(f"Generated forecasts from {name} forecaster")
            except Exception as e:
                print(f"Error generating forecasts from {name} forecaster: {str(e)}")
        
        # Combine forecasts
        if not forecaster_predictions:
            print("Warning: No forecasts available to combine")
            return {}
            
        # Get common categories
        all_categories = set()
        for forecaster_name, predictions in forecaster_predictions.items():
            all_categories.update(predictions.keys())
            
        if categories is not None:
            all_categories = all_categories.intersection(set(categories))
            
        # Process each category
        for category in all_categories:
            # Collect forecasts for this category
            category_forecasts = {}
            
            for forecaster_name, predictions in forecaster_predictions.items():
                if category in predictions:
                    category_forecasts[forecaster_name] = predictions[category]
            
            if not category_forecasts:
                continue
                
            # Align date indexes and create a common date range
            date_ranges = []
            for df in category_forecasts.values():
                if 'ds' in df.columns:
                    date_ranges.append(pd.DatetimeIndex(df['ds']))
                    
            if not date_ranges:
                continue
                
            # Use the shortest date range to avoid extrapolation issues
            min_len = min(len(dates) for dates in date_ranges)
            common_dates = sorted(set.intersection(*[set(dates[:min_len]) for dates in date_ranges]))
            
            if not common_dates:
                continue
                
            # Combine forecasts on common dates
            combined_forecasts = []
            
            for date in common_dates:
                # Collect values for this date from all forecasters
                date_values = []
                
                for forecaster_name, df in category_forecasts.items():
                    date_df = df[df['ds'] == date]
                    
                    if not date_df.empty:
                        value_col = 'amount' if 'amount' in date_df.columns else 'value'
                        
                        if value_col in date_df.columns:
                            value = date_df[value_col].values[0]
                            weight = self.weights.get(forecaster_name, 1.0)
                            date_values.append((value, weight, forecaster_name))
                
                if not date_values:
                    continue
                    
                # Combine values based on the specified method
                if combination_method == 'weighted_average':
                    total_weight = sum(weight for _, weight, _ in date_values)
                    combined_value = sum(value * weight for value, weight, _ in date_values) / total_weight
                elif combination_method == 'simple_average':
                    combined_value = sum(value for value, _, _ in date_values) / len(date_values)
                elif combination_method == 'median':
                    combined_value = np.median([value for value, _, _ in date_values])
                else:
                    # Default to weighted average
                    total_weight = sum(weight for _, weight, _ in date_values)
                    combined_value = sum(value * weight for value, weight, _ in date_values) / total_weight
                
                # Calculate lower and upper bounds (if available)
                lower_bounds = []
                upper_bounds = []
                
                for forecaster_name, df in category_forecasts.items():
                    date_df = df[df['ds'] == date]
                    
                    if not date_df.empty and 'lower_bound' in date_df.columns and 'upper_bound' in date_df.columns:
                        lower_bound = date_df['lower_bound'].values[0]
                        upper_bound = date_df['upper_bound'].values[0]
                        weight = self.weights.get(forecaster_name, 1.0)
                        
                        lower_bounds.append((lower_bound, weight))
                        upper_bounds.append((upper_bound, weight))
                
                # Calculate combined bounds
                if lower_bounds and upper_bounds:
                    if combination_method == 'weighted_average':
                        total_weight = sum(weight for _, weight in lower_bounds)
                        combined_lower = sum(value * weight for value, weight in lower_bounds) / total_weight
                        combined_upper = sum(value * weight for value, weight in upper_bounds) / total_weight
                    elif combination_method == 'simple_average':
                        combined_lower = sum(value for value, _ in lower_bounds) / len(lower_bounds)
                        combined_upper = sum(value for value, _ in upper_bounds) / len(upper_bounds)
                    elif combination_method == 'median':
                        combined_lower = np.median([value for value, _ in lower_bounds])
                        combined_upper = np.median([value for value, _ in upper_bounds])
                    else:
                        # Default to weighted average
                        total_weight = sum(weight for _, weight in lower_bounds)
                        combined_lower = sum(value * weight for value, weight in lower_bounds) / total_weight
                        combined_upper = sum(value * weight for value, weight in upper_bounds) / total_weight
                else:
                    combined_lower = None
                    combined_upper = None
                
                # Add combined forecast
                forecast_row = {
                    'ds': date,
                    'amount': combined_value
                }
                
                if combined_lower is not None and combined_upper is not None:
                    forecast_row['lower_bound'] = combined_lower
                    forecast_row['upper_bound'] = combined_upper
                
                combined_forecasts.append(forecast_row)
            
            if combined_forecasts:
                # Create DataFrame with combined forecasts
                ensemble_df = pd.DataFrame(combined_forecasts)
                self.forecasts[category] = ensemble_df
                
                print(f"Generated ensemble forecast for category '{category}'")
        
        return self.forecasts
    
    def save_model(self, output_dir: str = 'app/models/forecasting/saved_models') -> None:
        """
        Save all forecasters in the ensemble.
        
        Args:
            output_dir: Directory to save the models
        """
        if not self.is_fitted:
            raise ValueError("No fitted models to save")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each forecaster
        for name, forecaster in self.forecasters.items():
            if forecaster.is_fitted:
                try:
                    # Create a subdirectory for each forecaster
                    forecaster_dir = os.path.join(output_dir, name)
                    os.makedirs(forecaster_dir, exist_ok=True)
                    
                    # Save forecaster
                    forecaster.save_model(forecaster_dir)
                    
                    print(f"Saved {name} forecaster models to {forecaster_dir}")
                except Exception as e:
                    print(f"Error saving {name} forecaster: {str(e)}")
        
        # Save ensemble metadata
        ensemble_meta = {
            'weights': self.weights,
            'model_params': self.model_params,
            'forecaster_names': list(self.forecasters.keys())
        }
        
        meta_path = os.path.join(output_dir, f"{self.name}_metadata.pkl")
        joblib.dump(ensemble_meta, meta_path)
        
        print(f"Saved ensemble metadata to {meta_path}")
    
    def load_model(self, input_dir: str = 'app/models/forecasting/saved_models') -> None:
        """
        Load all forecasters in the ensemble.
        
        Args:
            input_dir: Directory to load the models from
        """
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory '{input_dir}' does not exist")
            
        # Load ensemble metadata
        meta_path = os.path.join(input_dir, f"{self.name}_metadata.pkl")
        
        if not os.path.exists(meta_path):
            raise ValueError(f"Ensemble metadata not found at {meta_path}")
            
        ensemble_meta = joblib.load(meta_path)
        
        # Update ensemble metadata
        self.weights = ensemble_meta['weights']
        self.model_params = ensemble_meta['model_params']
        
        # Initialize forecasters (if not already initialized)
        forecaster_names = ensemble_meta['forecaster_names']
        
        for name in forecaster_names:
            if name not in self.forecasters:
                if name == 'arima':
                    self.forecasters[name] = ARIMAForecaster()
                elif name == 'prophet':
                    self.forecasters[name] = ProphetForecaster()
                # Removed LSTM initialization
                else:
                    print(f"Warning: Unknown forecaster type '{name}'")
                    continue
            
            # Load forecaster models
            forecaster_dir = os.path.join(input_dir, name)
            
            if os.path.exists(forecaster_dir):
                try:
                    self.forecasters[name].load_model(forecaster_dir)
                    print(f"Loaded {name} forecaster models from {forecaster_dir}")
                except Exception as e:
                    print(f"Error loading {name} forecaster models: {str(e)}")
            else:
                print(f"Warning: Directory not found for {name} forecaster: {forecaster_dir}")
        
        # Update fitted status
        self.is_fitted = any(forecaster.is_fitted for forecaster in self.forecasters.values())
        
        if self.is_fitted:
            print(f"Successfully loaded ensemble forecaster with {len([f for f in self.forecasters.values() if f.is_fitted])} fitted component forecasters")
        else:
            print("Failed to load any fitted forecasters in the ensemble")