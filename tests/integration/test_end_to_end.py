"""
End-to-end integration tests for the Financial Health Assistant
"""
import os
import sys
import pandas as pd
import numpy as np
import pytest
import time
from datetime import datetime, timedelta

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.data.processor import TransactionProcessor
from app.models.categorization.nlp_categorizer import TransactionCategorizer
from app.models.time_series.time_series_processor import TimeSeriesProcessor
from app.models.forecasting.spending_forecaster import SpendingForecaster
from app.models.recommendation.recommendation_engine import RecommendationEngine

@pytest.fixture
def sample_raw_transactions():
    """Create sample raw transaction data that needs processing"""
    # Create 200 transactions over the last 6 months with various formats and issues
    start_date = datetime.now() - timedelta(days=180)
    dates = pd.date_range(start=start_date, periods=200, freq='D')
    
    np.random.seed(42)
    
    data = {
        'DATE': dates.strftime('%Y-%m-%d').tolist(),  # Date in string format
        'TRANSACTION DETAILS': [],
        ' WITHDRAWAL AMT ': [],  # Note the spaces in column names
        ' DEPOSIT AMT ': []
    }
    
    # Transaction types and common descriptions
    transaction_types = {
        'food': ['GROCERY STORE', 'RESTAURANT', 'FAST FOOD', 'CAFE'],
        'transport': ['GAS STATION', 'UBER', 'LYFT', 'CAR PAYMENT'],
        'housing': ['RENT', 'MORTGAGE', 'HOME INSURANCE'],
        'utilities': ['ELECTRIC BILL', 'WATER BILL', 'INTERNET', 'PHONE'],
        'entertainment': ['MOVIE', 'NETFLIX', 'SPOTIFY', 'GAMING'],
        'shopping': ['AMAZON', 'WALMART', 'TARGET', 'MACY\'S'],
        'health': ['DOCTOR', 'PHARMACY', 'HEALTH INSURANCE'],
        'income': ['SALARY', 'DIRECT DEPOSIT', 'PAYMENT RECEIVED']
    }
    
    # Generate transactions
    for _ in range(len(dates)):
        # Randomly select transaction type
        if np.random.random() < 0.1:  # 10% chance of income
            tx_type = 'income'
        else:
            tx_type = np.random.choice(list(transaction_types.keys()))
        
        # Generate transaction description
        if tx_type == 'income':
            description = np.random.choice(transaction_types[tx_type])
            withdrawal = 0.0
            deposit = np.random.uniform(2000, 5000)
        else:
            description = np.random.choice(transaction_types[tx_type])
            # Set realistic amount ranges by category
            if tx_type == 'housing':
                withdrawal = np.random.uniform(800, 2000)
            elif tx_type == 'food':
                withdrawal = np.random.uniform(10, 200)
            elif tx_type == 'transport':
                withdrawal = np.random.uniform(20, 150)
            elif tx_type == 'utilities':
                withdrawal = np.random.uniform(50, 300)
            elif tx_type == 'entertainment':
                withdrawal = np.random.uniform(10, 100)
            elif tx_type == 'shopping':
                withdrawal = np.random.uniform(20, 300)
            elif tx_type == 'health':
                withdrawal = np.random.uniform(20, 500)
            else:
                withdrawal = np.random.uniform(10, 100)
                
            deposit = 0.0
            
        # Add the transaction
        data['TRANSACTION DETAILS'].append(description)
        data[' WITHDRAWAL AMT '].append(withdrawal)
        data[' DEPOSIT AMT '].append(deposit)
    
    # Introduce some missing values and inconsistencies
    # 5% of transactions have missing descriptions
    missing_indices = np.random.choice(len(dates), size=int(0.05 * len(dates)), replace=False)
    for idx in missing_indices:
        data['TRANSACTION DETAILS'][idx] = np.nan
    
    # 3% of transactions have $0 in both withdrawal and deposit (data errors)
    zero_indices = np.random.choice(len(dates), size=int(0.03 * len(dates)), replace=False)
    for idx in zero_indices:
        data[' WITHDRAWAL AMT '][idx] = 0.0
        data[' DEPOSIT AMT '][idx] = 0.0
    
    # Convert to DataFrame
    return pd.DataFrame(data)

def test_complete_e2e_pipeline(sample_raw_transactions, tmp_path):
    """Test the complete end-to-end financial analysis pipeline"""
    print("\n=== Starting End-to-End Financial Health Assistant Test ===")
    
    # Set up working directories
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    
    for directory in [raw_dir, processed_dir, models_dir, results_dir]:
        directory.mkdir(exist_ok=True)
    
    # Step 1: Save raw data
    raw_file = raw_dir / "raw_transactions.csv"
    sample_raw_transactions.to_csv(raw_file, index=False)
    print(f"✓ Saved raw transaction data with {len(sample_raw_transactions)} records")
    
    # Step 2: Process transactions
    # Track performance
    start_time = time.time()
    
    processor = TransactionProcessor()
    raw_df = processor.load_transactions(raw_file)
    
    # Clean the data
    cleaned_df = processor.clean_transactions(raw_df)
    print(f"✓ Cleaned transactions: {len(cleaned_df)} records after cleaning")
    
    # Categorize transactions
    categorized_df = processor.simple_categorize(cleaned_df)
    
    # Calculate category distribution
    category_counts = categorized_df['category'].value_counts()
    print("✓ Initial categorization complete")
    print(f"  Category distribution: {dict(category_counts)}")
    
    # Save processed data
    processed_file = processed_dir / "processed_transactions.csv"
    categorized_df.to_csv(processed_file, index=False)
    
    processing_time = time.time() - start_time
    print(f"✓ Transaction processing completed in {processing_time:.2f} seconds")
    
    # Step 3: Train categorization model
    start_time = time.time()
    
    categorizer = TransactionCategorizer()
    descriptions = categorized_df['description'].tolist()
    categories = categorized_df['category'].tolist()
    
    accuracy = categorizer.train(descriptions, categories)
    print(f"✓ Trained categorization model with accuracy: {accuracy:.4f}")
    
    # Save the model
    categorizer_path = models_dir / "categorizer.joblib"
    categorizer.save_model(categorizer_path)
    print(f"✓ Saved categorization model")
    
    categorization_time = time.time() - start_time
    print(f"✓ Model training completed in {categorization_time:.2f} seconds")
    
    # Step 4: Generate time series data
    start_time = time.time()
    
    time_series_processor = TimeSeriesProcessor()
    time_series_data = time_series_processor.transactions_to_time_series(
        categorized_df,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category'
    )
    print(f"✓ Generated time series data for {len(time_series_data)} categories")
    
    # Save time series data
    for category, ts_df in time_series_data.items():
        ts_file = processed_dir / f"time_series_{category}.csv"
        ts_df.to_csv(ts_file, index=False)
    
    time_series_time = time.time() - start_time
    print(f"✓ Time series processing completed in {time_series_time:.2f} seconds")
    
    # Step 5: Train forecasting models
    start_time = time.time()
    
    forecaster = SpendingForecaster()
    forecaster.train_models(time_series_data, column='amount')
    print(f"✓ Trained forecasting models for {len(forecaster.models)} categories")
    
    # Save forecasting models
    forecasting_dir = models_dir / "forecasting"
    forecasting_dir.mkdir(exist_ok=True)
    forecaster.save_models(str(forecasting_dir))
    print(f"✓ Saved forecasting models")
    
    # Generate forecasts
    forecasts = forecaster.forecast(days=30)
    print(f"✓ Generated {len(forecasts)} category forecasts for the next 30 days")
    
    # Save forecasts
    for category, forecast_df in forecasts.items():
        forecast_file = results_dir / f"forecast_{category}.csv"
        forecast_df.to_csv(forecast_file, index=False)
    
    forecasting_time = time.time() - start_time
    print(f"✓ Forecasting completed in {forecasting_time:.2f} seconds")
    
    # Step 6: Generate recommendations
    start_time = time.time()
    
    engine = RecommendationEngine()
    recommendations = engine.generate_recommendations(
        categorized_df,
        forecasts=forecasts,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category',
        desc_col='description',
        limit=10
    )
    print(f"✓ Generated {len(recommendations)} financial recommendations")
    
    # Save recommendations
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_file = results_dir / "recommendations.csv"
    recommendations_df.to_csv(recommendations_file, index=False)
    
    # Print top 3 recommendations
    print("\nTop Financial Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec['message']} (Priority: {rec['priority']:.2f}, Savings: ${rec['savings_potential']:.2f})")
    
    # Visualize recommendations
    viz_file = results_dir / "recommendations_priority.png"
    engine.visualize_recommendations(recommendations, str(viz_file))
    print(f"✓ Created recommendation visualization")
    
    recommendation_time = time.time() - start_time
    print(f"✓ Recommendation generation completed in {recommendation_time:.2f} seconds")
    
    # Step 7: Verify results
    assert os.path.exists(processed_file), "Processed transactions file should exist"
    assert os.path.exists(categorizer_path), "Categorizer model file should exist"
    assert len(list(forecasting_dir.glob("*.joblib"))) > 0, "Forecasting model files should exist"
    assert os.path.exists(recommendations_file), "Recommendations file should exist"
    assert os.path.exists(viz_file), "Visualization file should exist"
    
    # Calculate total processing time
    total_time = processing_time + categorization_time + time_series_time + forecasting_time + recommendation_time
    print(f"\n=== Complete E2E Pipeline Execution Time: {total_time:.2f} seconds ===")
    print(f"  Data Processing:  {processing_time:.2f}s ({processing_time/total_time*100:.1f}%)")
    print(f"  Categorization:   {categorization_time:.2f}s ({categorization_time/total_time*100:.1f}%)")
    print(f"  Time Series:      {time_series_time:.2f}s ({time_series_time/total_time*100:.1f}%)")
    print(f"  Forecasting:      {forecasting_time:.2f}s ({forecasting_time/total_time*100:.1f}%)")
    print(f"  Recommendations:  {recommendation_time:.2f}s ({recommendation_time/total_time*100:.1f}%)")
    
    print("\n=== End-to-End Test Completed Successfully ===")