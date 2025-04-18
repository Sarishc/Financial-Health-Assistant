# app/models/forecasting/base_forecaster.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os

class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    Defines the interface that all forecaster models should implement.
    """
    
    def __init__(self, name: str = "base"):
        """
        Initialize the base forecaster.
        
        Args:
            name: Name of the forecaster
        """
        self.name = name
        self.models = {}  # Store models for different categories/time series
        self.is_fitted = False
        self.forecasts = {}
        self.forecast_horizon = 30  # Default forecast horizon (days)
        self.model_params = {}  # Model parameters
        self.visualization_dir = 'notebooks/visualizations/forecasts'
        os.makedirs(self.visualization_dir, exist_ok=True)
    
    @abstractmethod
    def fit(self, time_series_data: Dict[str, pd.DataFrame], **kwargs) -> None:
        """
        Fit the forecasting model on the time series data.
        
        Args:
            time_series_data: Dictionary mapping category names to time series DataFrames
            **kwargs: Additional parameters for the specific model implementation
        """
        pass
    
    @abstractmethod
    def predict(self, horizon: int = 30, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for the specified horizon.
        
        Args:
            horizon: Number of future time periods to forecast
            **kwargs: Additional parameters for the specific model implementation
            
        Returns:
            Dictionary mapping category names to DataFrames with forecasts
        """
        pass
    
    def evaluate(self, test_data: Dict[str, pd.DataFrame], metrics: List[str] = ['mae', 'rmse', 'mape']) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model performance on test data.
        
        Args:
            test_data: Dictionary mapping category names to test DataFrames
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of evaluation metrics for each category
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        results = {}
        
        for category, test_df in test_data.items():
            if category not in self.models:
                continue
                
            # Generate predictions for the test period
            horizon = len(test_df)
            predictions = self.predict(horizon=horizon, categories=[category])
            
            if category not in predictions:
                continue
                
            pred_df = predictions[category]
            
            # Calculate metrics
            category_metrics = {}
            
            # Align the time index
            actual_values = test_df['value'].values if 'value' in test_df.columns else test_df['amount'].values
            pred_values = pred_df['value'].values if 'value' in pred_df.columns else pred_df['amount'].values
            
            # Truncate to the shorter length if needed
            min_len = min(len(actual_values), len(pred_values))
            actual_values = actual_values[:min_len]
            pred_values = pred_values[:min_len]
            
            if 'mae' in metrics:
                category_metrics['mae'] = np.mean(np.abs(actual_values - pred_values))
                
            if 'rmse' in metrics:
                category_metrics['rmse'] = np.sqrt(np.mean((actual_values - pred_values) ** 2))
                
            if 'mape' in metrics and np.all(actual_values != 0):
                category_metrics['mape'] = np.mean(np.abs((actual_values - pred_values) / actual_values)) * 100
            
            results[category] = category_metrics
            
        return results
    
    def plot_forecast(self, category: str, historical_data: Optional[pd.DataFrame] = None, 
                     ax: Optional[plt.Axes] = None, title: Optional[str] = None) -> plt.Figure:
        """
        Plot forecasts for a specific category.
        
        Args:
            category: Category name
            historical_data: Historical data for comparison (optional)
            ax: Matplotlib axes to plot on (optional)
            title: Plot title (optional)
            
        Returns:
            Matplotlib figure with the plot
        """
        if not self.is_fitted or category not in self.forecasts:
            raise ValueError(f"No forecasts available for category '{category}'")
            
        forecast_df = self.forecasts[category]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure
            
        # Plot historical data if provided
        if historical_data is not None:
            ax.plot(historical_data.index, historical_data['value'] if 'value' in historical_data.columns else historical_data['amount'],
                   'k-', label='Historical')
        
        # Plot forecast
        ax.plot(forecast_df.index, forecast_df['value'] if 'value' in forecast_df.columns else forecast_df['amount'],
               'b-', label=f'{self.name} Forecast')
        
        # Plot confidence intervals if available
        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
            ax.fill_between(forecast_df.index, 
                           forecast_df['lower_bound'], 
                           forecast_df['upper_bound'],
                           color='b', alpha=0.2, label='95% Confidence Interval')
        
        # Set title and labels
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{self.name} Forecast for {category}')
            
        ax.set_xlabel('Date')
        ax.set_ylabel('Amount')
        ax.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def save_forecasts(self, output_dir: str = 'app/data/processed/forecasts') -> None:
        """
        Save forecasts to CSV files.
        
        Args:
            output_dir: Directory to save the forecasts
        """
        if not self.is_fitted or not self.forecasts:
            raise ValueError("No forecasts available to save")
            
        os.makedirs(output_dir, exist_ok=True)
        
        for category, forecast_df in self.forecasts.items():
            # Create safe filename
            safe_category = category.replace(' ', '_').lower()
            output_path = os.path.join(output_dir, f"{self.name}_{safe_category}_forecast.csv")
            forecast_df.to_csv(output_path)
            
        print(f"Saved {len(self.forecasts)} forecasts to {output_dir}")
    
    def save_model(self, output_dir: str = 'app/models/forecasting/saved_models') -> None:
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save the models
        """
        raise NotImplementedError("save_model method must be implemented by subclass")
    
    def load_model(self, input_dir: str = 'app/models/forecasting/saved_models') -> None:
        """
        Load trained models from disk.
        
        Args:
            input_dir: Directory to load the models from
        """
        raise NotImplementedError("load_model method must be implemented by subclass")