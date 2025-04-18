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

def load_forecasting_models(model_dir: str, freq: str = 'weekly'):
    """
    Load trained forecasting models.
    
    Args:
        model_dir: Directory containing saved models
        freq: Frequency ('daily', 'weekly', 'monthly')
        
    Returns:
        List of loaded forecaster objects
    """
    forecasters = []
    
    # Try to load ARIMA model
    arima_dir = os.path.join(model_dir, f"arima_{freq}")
    if os.path.exists(arima_dir):
        try:
            arima = ARIMAForecaster()
            arima.load_model(arima_dir)
            forecasters.append(arima)
            print(f"Loaded ARIMA model from {arima_dir}")
        except Exception as e:
            print(f"Error loading ARIMA model: {str(e)}")
    
    # Try to load Prophet model
    prophet_dir = os.path.join(model_dir, f"prophet_{freq}")
    if os.path.exists(prophet_dir):
        try:
            prophet = ProphetForecaster()
            prophet.load_model(prophet_dir)
            forecasters.append(prophet)
            print(f"Loaded Prophet model from {prophet_dir}")
        except Exception as e:
            print(f"Error loading Prophet model: {str(e)}")
    
    # Try to load LSTM model
    lstm_dir = os.path.join(model_dir, f"lstm_{freq}")
    if os.path.exists(lstm_dir):
        try:
            lstm = LSTMForecaster()
            lstm.load_model(lstm_dir)
            forecasters.append(lstm)
            print(f"Loaded LSTM model from {lstm_dir}")
        except Exception as e:
            print(f"Error loading LSTM model: {str(e)}")
    
    # Try to load Ensemble model
    ensemble_dir = os.path.join(model_dir, f"ensemble_{freq}")
    if os.path.exists(ensemble_dir):
        try:
            ensemble = EnsembleForecaster()
            ensemble.load_model(ensemble_dir)
            forecasters.append(ensemble)
            print(f"Loaded Ensemble model from {ensemble_dir}")
        except Exception as e:
            print(f"Error loading Ensemble model: {str(e)}")
    
    return forecasters

def evaluate_forecasting_models(forecasters: list, test_data: dict, historical_data: dict):
    """
    Evaluate the performance of forecasting models on test data.
    
    Args:
        forecasters: List of forecaster objects to evaluate
        test_data: Dictionary mapping category names to test DataFrames
        historical_data: Dictionary mapping category names to historical DataFrames
    """
    if not forecasters:
        print("No forecasters to evaluate")
        return
    
    # Initialize evaluator
    evaluator = ForecastEvaluator()
    
    # Generate forecasts
    for forecaster in forecasters:
        horizon = max(len(df) for df in test_data.values())
        forecaster.predict(horizon=horizon)
    
    # Compare forecasters
    print("\nComparing forecasters...")
    evaluation_results = evaluator.compare_forecasters(
        forecasters,
        test_data
    )
    
    # Print summary of results
    print("\nEvaluation Results:")
    for forecaster_name, category_results in evaluation_results.items():
        print(f"\n{forecaster_name}:")
        for category, metrics in category_results.items():
            print(f"  {category}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
    
    # Determine best forecaster for each metric
    for metric in ['mae', 'rmse', 'mape']:
        best_forecasters = evaluator.get_best_forecaster(metric=metric)
        
        print(f"\nBest forecaster by {metric.upper()}:")
        for category, forecaster_name in best_forecasters.items():
            print(f"  {category}: {forecaster_name}")
    
    # Create evaluation report
    print("\nCreating evaluation report...")
    report_path = evaluator.create_evaluation_report(
        forecasters,
        test_data,
        historical_data
    )
    
    print(f"Evaluation report saved to: {report_path}")

def prepare_test_data(time_series_file: str, test_size: float = 0.2):
    """
    Prepare test data from time series file.
    
    Args:
        time_series_file: Path to time series file
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (historical_data, test_data) dictionaries
    """
    # Load time series data
    df = pd.read_csv(time_series_file)
    
    # Ensure date column is datetime
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Prepare data by category
    historical_data = {}
    test_data = {}
    
    if 'category' in df.columns:
        # Prepare data for each category
        categories = df['category'].unique()
        
        for category in categories:
            category_df = df[df['category'] == category].copy()
            if len(category_df) >= 10:  # Need at least 10 data points
                # Sort by date
                sorted_df = category_df.sort_values('transaction_date')
                
                # Calculate split point
                split_idx = int(len(sorted_df) * (1 - test_size))
                
                # Split data
                historical_data[category] = sorted_df.iloc[:split_idx][['transaction_date', 'amount_sum']].copy()
                test_data[category] = sorted_df.iloc[split_idx:][['transaction_date', 'amount_sum']].copy()
                
                print(f"Category '{category}': {len(historical_data[category])} historical samples, {len(test_data[category])} test samples")
    else:
        # No category column, use 'all' as the category
        # Sort by date
        sorted_df = df.sort_values('transaction_date')
        
        # Calculate split point
        split_idx = int(len(sorted_df) * (1 - test_size))
        
        # Split data
        historical_data['all'] = sorted_df.iloc[:split_idx][['transaction_date', 'amount_sum']].copy()
        test_data['all'] = sorted_df.iloc[split_idx:][['transaction_date', 'amount_sum']].copy()
        
        print(f"Category 'all': {len(historical_data['all'])} historical samples, {len(test_data['all'])} test samples")
    
    return historical_data, test_data

def main():
    """Main execution function"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate forecasting models")
    parser.add_argument('--freq', type=str, default='weekly', choices=['daily', 'weekly', 'monthly'],
                      help="Time series frequency to evaluate")
    parser.add_argument('--model-dir', type=str, default='app/models/forecasting/saved_models',
                      help="Directory containing saved models")
    parser.add_argument('--ts-dir', type=str, default='app/data/processed/time_series',
                      help="Directory containing time series data")
    args = parser.parse_args()
    
    # Check if time series data exists
    time_series_file = os.path.join(args.ts_dir, f"{args.freq}_ts.csv")
    
    if not os.path.exists(time_series_file):
        print(f"Error: Time series file not found at {time_series_file}")
        sys.exit(1)
    
    # Prepare test data
    print(f"Preparing test data from {time_series_file}...")
    historical_data, test_data = prepare_test_data(time_series_file)
    
    if not test_data:
        print("Error: No test data available")
        sys.exit(1)
    
    # Load forecasting models
    print(f"\nLoading forecasting models for {args.freq} frequency...")
    forecasters = load_forecasting_models(args.model_dir, args.freq)
    
    if not forecasters:
        print(f"Error: No forecasting models found in {args.model_dir}")
        sys.exit(1)
    
    # Evaluate models
    print("\nEvaluating forecasting models...")
    evaluate_forecasting_models(forecasters, test_data, historical_data)

if __name__ == "__main__":
    main()