import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Only import ARIMA and Prophet forecasters
from app.models.forecasting.arima_forecaster import ARIMAForecaster
from app.models.forecasting.prophet_forecaster import ProphetForecaster
from app.models.forecasting.ensemble_forecaster import EnsembleForecaster
from app.models.forecasting.forecast_evaluator import ForecastEvaluator

def execute_day5_tasks():
    """Execute all Day 5 tasks for the Financial Health Assistant project"""
    
    print("=" * 80)
    print("FINANCIAL HEALTH ASSISTANT - DAY 5: SPENDING FORECASTING MODEL")
    print("=" * 80)
    
    # Step 1: Create necessary directories
    os.makedirs('app/models/forecasting', exist_ok=True)
    os.makedirs('app/models/forecasting/saved_models', exist_ok=True)
    os.makedirs('app/data/processed/forecasts', exist_ok=True)
    os.makedirs('notebooks/visualizations/forecasts', exist_ok=True)
    
    # Step 2: Check if time series data exists
    ts_dir = 'app/data/processed/time_series'
    time_series_files = [
        os.path.join(ts_dir, 'daily_ts.csv'),
        os.path.join(ts_dir, 'weekly_ts.csv'),
        os.path.join(ts_dir, 'monthly_ts.csv')
    ]
    
    missing_files = [f for f in time_series_files if not os.path.exists(f)]
    
    if missing_files:
        print("Warning: The following time series files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("Running process_time_series.py to generate them...")
        
        # Run process_time_series.py
        from scripts.process_time_series import process_transaction_time_series
        process_transaction_time_series()
    
    # Step 3: Load time series data
    print("\nLoading time series data...")
    time_series_data = {}
    
    for freq, file in zip(['daily', 'weekly', 'monthly'], time_series_files):
        if os.path.exists(file):
            time_series_data[freq] = pd.read_csv(file)
            print(f"Loaded {freq} time series with {len(time_series_data[freq])} rows")
    
    if not time_series_data:
        print("Error: No time series data available. Cannot continue.")
        return False
    
    # Step 4: Train forecasting models
    print("\nTraining forecasting models...")
    
    # Use the model with the best balance of granularity and sample size
    # Usually this is weekly or monthly data
    if 'weekly' in time_series_data:
        primary_data = 'weekly'
    elif 'monthly' in time_series_data:
        primary_data = 'monthly'
    else:
        primary_data = list(time_series_data.keys())[0]
        
    print(f"Using {primary_data} data as the primary time series for forecasting")
    
    # Prepare data by category
    df = time_series_data[primary_data]
    category_data = {}
    
    if 'category' in df.columns:
        # Prepare data for each category
        categories = df['category'].unique()
        
        for category in categories:
            category_df = df[df['category'] == category].copy()
            if len(category_df) >= 10:  # Need at least 10 data points
                category_data[category] = category_df[['transaction_date', 'amount_sum']]
    else:
        # No category column, use 'all' as the category
        category_data['all'] = df[['transaction_date', 'amount_sum']].copy()
    
    # Split into train and test sets
    train_data = {}
    test_data = {}
    test_size = 0.2
    
    for category, cat_df in category_data.items():
        # Sort by date
        sorted_df = cat_df.sort_values('transaction_date')
        
        # Calculate split point
        split_idx = int(len(sorted_df) * (1 - test_size))
        
        # Split data
        train_data[category] = sorted_df.iloc[:split_idx].copy()
        test_data[category] = sorted_df.iloc[split_idx:].copy()
        
        print(f"Category '{category}': {len(train_data[category])} train samples, {len(test_data[category])} test samples")
    
    # Train ARIMA models
    print("\nTraining ARIMA models...")
    arima = ARIMAForecaster()
    arima.fit(train_data, date_col='transaction_date', value_col='amount_sum')
    
    # Generate ARIMA forecasts (30 days ahead)
    arima_forecasts = arima.predict(horizon=30)
    arima.save_forecasts(os.path.join('app/data/processed/forecasts', f"arima_{primary_data}"))
    arima.save_model(os.path.join('app/models/forecasting/saved_models', f"arima_{primary_data}"))
    
    # Train Prophet models
    print("\nTraining Prophet models...")
    
    try:
        prophet = ProphetForecaster()
        prophet.fit(train_data, date_col='transaction_date', value_col='amount_sum')
        
        # Generate Prophet forecasts
        prophet_forecasts = prophet.predict(horizon=30)
        prophet.save_forecasts(os.path.join('app/data/processed/forecasts', f"prophet_{primary_data}"))
        prophet.save_model(os.path.join('app/models/forecasting/saved_models', f"prophet_{primary_data}"))
        
        # Create ensemble with ARIMA and Prophet
        print("\nCreating ensemble forecaster with ARIMA and Prophet...")
        ensemble = EnsembleForecaster()
        ensemble.add_forecaster(arima, weight=0.4)
        ensemble.add_forecaster(prophet, weight=0.6)
        
        # Generate ensemble forecasts
        ensemble_forecasts = ensemble.predict(horizon=30)
        ensemble.save_forecasts(os.path.join('app/data/processed/forecasts', f"ensemble_{primary_data}"))
        ensemble.save_model(os.path.join('app/models/forecasting/saved_models', f"ensemble_{primary_data}"))
        
        # Evaluate models
        print("\nEvaluating ARIMA, Prophet, and Ensemble models...")
        evaluator = ForecastEvaluator()
        
        # Try to evaluate forecasters
        try:
            # Compare forecasters
            evaluation_results = evaluator.compare_forecasters(
                [arima, prophet, ensemble],
                test_data
            )
            
            # Create evaluation report
            print("\nCreating evaluation report...")
            report_path = evaluator.create_evaluation_report(
                [arima, prophet, ensemble],
                test_data,
                train_data
            )
            
            print(f"Evaluation report saved to: {report_path}")
            
            # Get best forecasters
            best_forecasters = evaluator.get_best_forecaster(metric='rmse')
            
            print("\nBest forecasting model for each category (based on RMSE):")
            for category, forecaster in best_forecasters.items():
                print(f"  {category}: {forecaster}")
        except Exception as e:
            print(f"Warning: Error in evaluation: {str(e)}")
    
    except Exception as e:
        print(f"Error training Prophet models: {str(e)}")
        print("Continuing with ARIMA only")
    
    # Step 5: Generate basic forecast visualizations for the top categories
    print("\nGenerating basic forecast visualizations...")
    
    # Get the top categories by total amount
    if 'category' in df.columns:
        top_categories = df.groupby('category')['amount_sum'].sum().abs().sort_values(ascending=False).head(5).index.tolist()
    else:
        top_categories = ['all']
    
    # Create directory for visualizations
    viz_dir = 'notebooks/visualizations/forecasts/top_categories'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Use ARIMA forecasts which should be available
    for category in top_categories:
        if category in arima.forecasts:
            try:
                # Use non-interactive backend
                fig = plt.figure(figsize=(10, 6))
                
                # Get the forecast data
                forecast_df = arima.forecasts[category]
                
                # Plot the forecast
                plt.plot(pd.to_datetime(forecast_df['ds']), forecast_df['amount'], 'b-', label='ARIMA Forecast')
                
                # Add confidence intervals if available
                if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                    plt.fill_between(
                        pd.to_datetime(forecast_df['ds']),
                        forecast_df['lower_bound'],
                        forecast_df['upper_bound'],
                        color='b', alpha=0.2, label='95% Confidence Interval'
                    )
                
                # Add historical data if available
                if category in train_data:
                    plt.plot(
                        pd.to_datetime(train_data[category]['transaction_date']),
                        train_data[category]['amount_sum'],
                        'k-', label='Historical Data'
                    )
                
                # Add labels and legend
                plt.title(f'Forecast for {category}')
                plt.xlabel('Date')
                plt.ylabel('Amount')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save the plot
                output_path = os.path.join(viz_dir, f"{category.replace(' ', '_').lower()}_forecast.png")
                plt.savefig(output_path)
                plt.close(fig)
                
                print(f"Saved forecast visualization for {category} to {output_path}")
            except Exception as e:
                print(f"Error creating visualization for {category}: {str(e)}")
    
    print("\nDay 5 execution completed successfully!")
    return True

if __name__ == "__main__":
    success = execute_day5_tasks()
    
    if not success:
        print("Day 5 execution failed")
        sys.exit(1)