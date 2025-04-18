# tests/test_time_series.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.time_series.time_series_processor import TimeSeriesProcessor

def create_sample_data(n_days=100):
    """Create sample transaction data for testing"""
    np.random.seed(42)
    
    # Generate dates
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate random transactions (3 per day on average)
    n_transactions = n_days * 3
    transaction_dates = np.random.choice(dates, size=n_transactions)
    
    # Generate random categories
    categories = ['food', 'transport', 'shopping', 'utilities', 'entertainment']
    transaction_categories = np.random.choice(categories, size=n_transactions)
    
    # Generate random amounts (negative for expenses)
    amounts = -np.random.uniform(10, 200, size=n_transactions)
    
    # Add some income transactions (positive amounts)
    income_indices = np.random.choice(range(n_transactions), size=int(n_transactions*0.1))
    amounts[income_indices] = np.random.uniform(500, 2000, size=len(income_indices))
    
    # Create DataFrame
    df = pd.DataFrame({
        'transaction_date': transaction_dates,
        'category': transaction_categories,
        'amount': amounts
    })
    
    return df

def test_time_series_processor():
    """Test the time series processor functionality"""
    
    print("=" * 50)
    print("TESTING TIME SERIES PROCESSOR")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample data with {len(df)} transactions")
    
    # Initialize processor
    processor = TimeSeriesProcessor()
    
    # Test conversion to time series
    print("\nTesting conversion to time series...")

    daily_ts = processor.convert_to_time_series(
        df,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category',
        freq='D'
    )
    print(f"Created daily time series with {len(daily_ts)} records")
    print(daily_ts.head())
    
    # Test temporal feature extraction
    print("\nTesting temporal feature extraction...")
    ts_with_features = processor.extract_temporal_features(daily_ts)
    print(f"Added {len(ts_with_features.columns) - len(daily_ts.columns)} temporal features")
    print("New features:", [col for col in ts_with_features.columns if col not in daily_ts.columns])
    
    # Test lag feature creation
    print("\nTesting lag feature creation...")
    ts_with_lags = processor.create_lagged_features(
        daily_ts,
        value_col='amount_sum',
        lag_periods=[1, 7],
        group_col='category'
    )
    print(f"Added {len(ts_with_lags.columns) - len(daily_ts.columns)} lag features")
    print("New features:", [col for col in ts_with_lags.columns if col not in daily_ts.columns])
    
    # Test rolling feature creation
    print("\nTesting rolling feature creation...")
    ts_with_rolling = processor.create_rolling_features(
        daily_ts,
        value_col='amount_sum',
        window_sizes=[7, 14],
        group_col='category'
    )
    print(f"Added {len(ts_with_rolling.columns) - len(daily_ts.columns)} rolling features")
    print("New features:", [col for col in ts_with_rolling.columns if col not in daily_ts.columns])
    
    # Test outlier detection
    print("\nTesting outlier detection...")
    ts_with_outliers = processor.detect_outliers(
        daily_ts,
        value_col='amount_sum',
        method='zscore',
        threshold=3.0
    )
    num_outliers = ts_with_outliers['amount_sum_is_outlier'].sum()
    print(f"Detected {num_outliers} outliers out of {len(ts_with_outliers)} records")
    
    # Test visualization
    print("\nTesting time series visualization...")
    try:
        fig = processor.visualize_time_series(
            daily_ts,
            date_col='transaction_date',
            value_col='amount_sum',
            category_col='category',
            title='Test Visualization'
        )
        print("Visualization created successfully")
        plt.close(fig)
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
    
    # Test full processing pipeline
    print("\nTesting full processing pipeline...")
    processed_ts = processor.process_time_series_data(
        df,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category',
        freq='D'
    )
    print(f"Processed time series has {len(processed_ts)} records and {len(processed_ts.columns)} features")
    
    print("\nTime series processor tests completed successfully!")
    return True

if __name__ == "__main__":
    test_time_series_processor()