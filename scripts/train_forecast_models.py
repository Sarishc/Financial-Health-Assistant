import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.forecasting.arima_forecaster import ARIMAForecaster
from app.models.forecasting.prophet_forecaster import ProphetForecaster
from app.models.forecasting.lstm_forecaster import LSTMForecaster
from app.models.forecasting.ensemble_forecaster import EnsembleForecaster
from app.models.forecasting.forecast_evaluator import ForecastEvaluator

def load_time_series_data(ts_dir: str = 'app/data/processed/time_series') -> dict:
    """
    Load time series data from files.
    
    Args:
        ts_dir: Directory containing time series data files
        
    Returns:
        Dictionary mapping frequency to DataFrames
    """
    time_series_data = {}
    
    # Check for time series files
    ts_files = {
        'daily': os.path.join(ts_dir, 'daily_ts.csv'),
        'weekly': os.path.join(ts_dir, 'weekly_ts.csv'),
        'monthly': os.path.join(ts_dir, 'monthly_ts.csv')
    }
    
    for freq, file_path in ts_files.items():
        if os.path.exists(file_path):
            print(f"Loading {freq} time series data from {file_path}")
            time_series_data[freq] = pd.read_csv(file_path)
            
            # Ensure date column is datetime
            if 'transaction_date' in time_series_data[freq].columns:
                time_series_data[freq]['transaction_date'] = pd.to_datetime(time_series_data[freq]['transaction_date'])
        else:
            print(f"Warning: {freq} time series file not found at {file_path}")
    
    return time_series_data

def prepare_data_by_category(df: pd.DataFrame, 
                           category_col: str = 'category',
                           date_col: str = 'transaction_date',
                           value_col: str = 'amount_sum') -> dict:
    """
    Prepare data by category for forecasting.
    
    Args:
        df: DataFrame containing time series data
        category_col: Column name for categories
        date_col: Column name for dates
        value_col: Column name for the target values
        
    Returns:
        Dictionary mapping category names to DataFrames
    """
    category_data = {}
    
    if category_col not in df.columns:
        # No category column, use 'all' as the category
        category_data['all'] = df[[date_col, value_col]].copy()
        return category_data
    
    # Get unique categories
    categories = df[category_col].unique()
    
    # Prepare data for each category
    for category in categories:
        category_df = df[df[category_col] == category].copy()
        category_data[category] = category_df[[date_col, value_col]]
    
    return category_data

def split_time_series(data: dict, test_size: float = 0.2) -> tuple:
    """
    Split time series data into train and test sets.
    
    Args:
        data: Dictionary mapping category names to DataFrames
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (train_data, test_data) dictionaries
    """
    train_data = {}
    test_data = {}
    
    for category, df in data.items():
        # Sort by date
        date_col = 'transaction_date' if 'transaction_date' in df.columns else 'ds'
        sorted_df = df.sort_values(date_col)
        
        # Calculate split point
        split_idx = int(len(sorted_df) * (1 - test_size))
        
        # Split data
        train_data[category] = sorted_df.iloc[:split_idx].copy()
        test_data[category] = sorted_df.iloc[split_idx:].copy()
        
        print(f"Category '{category}': {len(train_data[category])} train samples, {len(test_data[category])} test samples")
    
    return train_data, test_data

def train_forecasting_models(time_series_data: dict, forecast_horizon: int = 30, evaluate: bool = True):
    """
    Train forecasting models on time series data.
    
    Args:
        time_series_data: Dictionary mapping frequency to DataFrames
        forecast_horizon: Number of future time periods to forecast
        evaluate: Whether to evaluate models on test data
    """
    # Create directory for saved models
    model_dir = 'app/models/forecasting/saved_models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Create directory for forecasts
    forecast_dir = 'app/data/processed/forecasts'
    os.makedirs(forecast_dir, exist_ok=True)
    
    # Process each frequency
    for freq, df in time_series_data.items():
        print(f"\n{'='*80}")
        print(f"Processing {freq} time series data")
        print(f"{'='*80}")
        
        # Prepare data by category
        category_data = prepare_data_by_category(df)
        
        # Split into train and test sets
        train_data, test_data = split_time_series(category_data)
        
        # Initialize forecasters
        arima = ARIMAForecaster()
        prophet = ProphetForecaster()
        lstm = LSTMForecaster()
        
        # Train ARIMA models
        print("\nTraining ARIMA models...")
        arima.fit(train_data, date_col='transaction_date', value_col='amount_sum')
        
        # Generate ARIMA forecasts
        arima_forecasts = arima.predict(horizon=forecast_horizon)
        arima.save_forecasts(os.path.join(forecast_dir, f"arima_{freq}"))
        arima.save_model(os.path.join(model_dir, f"arima_{freq}"))
        
        # Train Prophet models
        print("\nTraining Prophet models...")
        prophet.fit(train_data, date_col='transaction_date', value_col='amount_sum')
        
        # Generate Prophet forecasts
        prophet_forecasts = prophet.predict(horizon=forecast_horizon)
        prophet.save_forecasts(os.path.join(forecast_dir, f"prophet_{freq}"))
        prophet.save_model(os.path.join(model_dir, f"prophet_{freq}"))
        
        # Train LSTM models (only if enough data)
        print("\nTraining LSTM models...")
        try:
            lstm.fit(train_data, date_col='transaction_date', value_col='amount_sum')
            
            # Generate LSTM forecasts
            lstm_forecasts = lstm.predict(horizon=forecast_horizon, monte_carlo_simulations=100)
            lstm.save_forecasts(os.path.join(forecast_dir, f"lstm_{freq}"))
            lstm.save_model(os.path.join(model_dir, f"lstm_{freq}"))
            
            # Create ensemble
            print("\nCreating ensemble forecaster...")
            ensemble = EnsembleForecaster()
            ensemble.add_forecaster(arima)
            ensemble.add_forecaster(prophet)
            ensemble.add_forecaster(lstm)
            
            # Generate ensemble forecasts
            ensemble_forecasts = ensemble.predict(horizon=forecast_horizon)
            ensemble.save_forecasts(os.path.join(forecast_dir, f"ensemble_{freq}"))
            ensemble.save_model(os.path.join(model_dir, f"ensemble_{freq}"))
            
            # Evaluate models if requested
            if evaluate and test_data:
                print("\nEvaluating models...")
                evaluator = ForecastEvaluator()
                
                # Compare forecasters
                evaluation_results = evaluator.compare_forecasters(
                    [arima, prophet, lstm, ensemble],
                    test_data
                )
                
                # Create evaluation report
                print("\nCreating evaluation report...")
                report_path = evaluator.create_evaluation_report(
                    [arima, prophet, lstm, ensemble],
                    test_data,
                    train_data
                )
                
                print(f"Evaluation report saved to: {report_path}")
                
        except Exception as e:
            print(f"Error training LSTM models: {str(e)}")
            
            # Create ensemble without LSTM
            print("\nCreating ensemble forecaster without LSTM...")
            ensemble = EnsembleForecaster()
            ensemble.add_forecaster(arima, weight=0.4)
            ensemble.add_forecaster(prophet, weight=0.6)
            
            # Generate ensemble forecasts
            ensemble_forecasts = ensemble.predict(horizon=forecast_horizon)
            ensemble.save_forecasts(os.path.join(forecast_dir, f"ensemble_{freq}"))
            ensemble.save_model(os.path.join(model_dir, f"ensemble_{freq}"))
            
            # Evaluate models if requested
            if evaluate and test_data:
                print("\nEvaluating models...")
                evaluator = ForecastEvaluator()
                
                # Compare forecasters
                evaluation_results = evaluator.compare_forecasters(
                    [arima, prophet],
                    test_data
                )
                
                # Create evaluation report
                print("\nCreating evaluation report...")
                report_path = evaluator.create_evaluation_report(
                    [arima, prophet],
                    test_data,
                    train_data
                )
                
                print(f"Evaluation report saved to: {report_path}")
        
        print(f"\nCompleted forecasting for {freq} time series")

if __name__ == "__main__":
    # Load time series data
    time_series_data = load_time_series_data()
    
    if not time_series_data:
        print("No time series data found. Please run process_time_series.py first.")
        sys.exit(1)
    
    # Train forecasting models
    train_forecasting_models(time_series_data, forecast_horizon=30)