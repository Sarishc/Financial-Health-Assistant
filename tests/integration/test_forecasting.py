"""
Integration tests for the forecasting components
"""
import os
import sys
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.models.forecasting.spending_forecaster import SpendingForecaster
from app.models.time_series.time_series_processor import TimeSeriesProcessor

@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing forecasting"""
    # Create date range for the past 3 years with daily data
    dates = pd.date_range(end=datetime.now(), periods=365*3, freq='D')
    
    # Create spending patterns with seasonality and trend
    np.random.seed(42)  # For reproducibility
    
    # Dictionary to hold time series data for each category
    time_series_dict = {}
    
    # Food category with weekly seasonality
    food_trend = np.linspace(30, 40, len(dates))  # Increasing trend
    food_weekly = 15 * (np.sin(np.arange(len(dates)) * (2 * np.pi / 7)) + 1)  # Weekly pattern
    food_noise = np.random.normal(0, 5, len(dates))
    food_values = food_trend + food_weekly + food_noise
    
    # Transportation with monthly seasonality
    transport_trend = np.linspace(100, 80, len(dates))  # Decreasing trend
    transport_monthly = 30 * (np.sin(np.arange(len(dates)) * (2 * np.pi / 30)) + 1)  # Monthly pattern
    transport_noise = np.random.normal(0, 10, len(dates))
    transport_values = transport_trend + transport_monthly + transport_noise
    
    # Housing with yearly seasonality
    housing_trend = np.linspace(500, 600, len(dates))  # Increasing trend
    housing_yearly = 50 * (np.sin(np.arange(len(dates)) * (2 * np.pi / 365)) + 1)  # Yearly pattern
    housing_noise = np.random.normal(0, 20, len(dates))
    housing_values = housing_trend + housing_yearly + housing_noise
    
    # Create DataFrames for each category
    time_series_dict['food'] = pd.DataFrame({
        'date': dates,
        'amount': food_values
    })
    
    time_series_dict['transport'] = pd.DataFrame({
        'date': dates,
        'amount': transport_values
    })
    
    time_series_dict['housing'] = pd.DataFrame({
        'date': dates,
        'amount': housing_values
    })
    
    return time_series_dict

def test_forecasting_integration(sample_time_series_data, tmp_path):
    """Test the forecasting system end-to-end"""
    # Initialize forecaster
    forecaster = SpendingForecaster()
    
    # Train models on sample data
    forecaster.train_models(sample_time_series_data, column='amount')
    
    # Verify models were created for each category
    assert len(forecaster.models) == len(sample_time_series_data), "Should create a model for each category"
    
    # Generate forecasts
    forecast_horizon = 30  # 30 days
    forecasts = forecaster.forecast(days=forecast_horizon)
    
    # Verify forecasts structure
    assert isinstance(forecasts, dict), "Forecasts should be returned as a dictionary"
    assert len(forecasts) == len(sample_time_series_data), "Should generate forecasts for each category"
    
    # Check forecast content for each category
    for category, forecast_df in forecasts.items():
        assert isinstance(forecast_df, pd.DataFrame), f"Forecast for {category} should be a DataFrame"
        assert len(forecast_df) == forecast_horizon, f"Forecast for {category} should have {forecast_horizon} rows"
        assert 'date' in forecast_df.columns, f"Forecast for {category} should have date column"
        assert 'forecast' in forecast_df.columns, f"Forecast for {category} should have forecast column"
        
        # Check that dates are in the future
        current_date = datetime.now().date()
        assert all(pd.to_datetime(date).date() > current_date for date in forecast_df['date']), \
            f"All forecast dates for {category} should be in the future"
    
    # Save and load models to test persistence
    model_dir = tmp_path / "forecast_models"
    model_dir.mkdir(exist_ok=True)
    
    # Save models
    forecaster.save_models(str(model_dir))
    
    # Check that model files exist
    model_files = list(model_dir.glob("*.joblib"))
    assert len(model_files) > 0, "Model files should be saved"
    
    # Create a new forecaster and load the saved models
    new_forecaster = SpendingForecaster()
    new_forecaster.load_models(str(model_dir))
    
    # Verify models loaded correctly
    assert len(new_forecaster.models) == len(forecaster.models), "Should load all saved models"
    
    # Generate forecasts with loaded models
    new_forecasts = new_forecaster.forecast(days=forecast_horizon)
    
    # Verify new forecasts structure
    assert len(new_forecasts) == len(forecasts), "New forecaster should generate same number of forecasts"
    
    # Compare forecasts to ensure they match (within numerical precision)
    for category in forecasts:
        assert category in new_forecasts, f"Category {category} should be in new forecasts"
        
        # Extract forecast values for comparison
        orig_values = forecasts[category]['forecast'].values
        new_values = new_forecasts[category]['forecast'].values
        
        # Check that forecasts are similar (allowing for minor numerical differences)
        np.testing.assert_allclose(orig_values, new_values, rtol=1e-5, atol=1e-5,
                                   err_msg=f"Forecasts for {category} should match after model reload")
    
    # Test visualization functionality (just check that it doesn't error)
    for category in forecasts:
        output_path = tmp_path / f"{category}_forecast.png"
        try:
            forecaster.visualize_forecast(
                category,
                forecasts[category],
                sample_time_series_data[category],
                column='amount',
                output_path=str(output_path)
            )
            assert output_path.exists(), f"Visualization file for {category} should be created"
        except Exception as e:
            pytest.fail(f"Visualization for {category} failed: {str(e)}")
    
    print("Integration test of forecasting completed successfully!")