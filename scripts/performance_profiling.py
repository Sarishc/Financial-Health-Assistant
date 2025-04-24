#!/usr/bin/env python
"""
Script for profiling performance of the Financial Health Assistant
"""
import os
import sys
import cProfile
import pstats
import io
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import importlib

# Add the project root directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def profile_module(module_name, function_name, *args, **kwargs):
    """
    Profile a specific function in a module
    
    Args:
        module_name: Name of the module
        function_name: Name of the function to profile
        args: Positional arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        
    Returns:
        Profile statistics
    """
    # Import the module
    module = importlib.import_module(module_name)
    
    # Get the function
    func = getattr(module, function_name)
    
    # Create a profile
    pr = cProfile.Profile()
    pr.enable()
    
    # Run the function
    result = func(*args, **kwargs)
    
    # Disable the profile
    pr.disable()
    
    return result, pr

def run_profiling(component, output_dir):
    """
    Run profiling for a specific component
    
    Args:
        component: The component to profile
        output_dir: Directory to save profiling results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if component == 'transaction_processing':
        profile_transaction_processing(output_dir)
    elif component == 'categorization':
        profile_categorization(output_dir)
    elif component == 'forecasting':
        profile_forecasting(output_dir)
    elif component == 'recommendations':
        profile_recommendations(output_dir)
    elif component == 'all':
        profile_transaction_processing(output_dir)
        profile_categorization(output_dir)
        profile_forecasting(output_dir)
        profile_recommendations(output_dir)
    else:
        print(f"Unknown component: {component}")
        sys.exit(1)

def profile_transaction_processing(output_dir):
    """Profile the transaction processing component"""
    print("\n=== Profiling Transaction Processing ===")
    
    # Import the necessary modules
    from app.data.processor import TransactionProcessor
    
    # Create a test dataset
    print("Generating test data...")
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(end=datetime.now(), periods=10000, freq='H')
    
    # Generate transaction data
    data = {
        'transaction_date': dates,
        'description': [f"Transaction {i}" for i in range(10000)],
        'amount': np.random.uniform(-1000, 1000, size=10000)
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV for testing file I/O
    csv_path = os.path.join(output_dir, "test_transactions.csv")
    df.to_csv(csv_path, index=False)
    
    # Profile each function
    processor = TransactionProcessor()
    
    # Profile load_transactions
    print("Profiling load_transactions...")
    _, pr_load = profile_module('app.data.processor', 'TransactionProcessor.load_transactions', 
                               processor, csv_path)
    
    # Save the profile stats
    s = io.StringIO()
    ps = pstats.Stats(pr_load, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    with open(os.path.join(output_dir, "profile_load_transactions.txt"), 'w') as f:
        f.write(s.getvalue())
    
    # Profile clean_transactions
    print("Profiling clean_transactions...")
    df_loaded = processor.load_transactions(csv_path)
    _, pr_clean = profile_module('app.data.processor', 'TransactionProcessor.clean_transactions', 
                                processor, df_loaded)
    
    # Save the profile stats
    s = io.StringIO()
    ps = pstats.Stats(pr_clean, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    with open(os.path.join(output_dir, "profile_clean_transactions.txt"), 'w') as f:
        f.write(s.getvalue())
    
    # Profile simple_categorize
    print("Profiling simple_categorize...")
    df_cleaned = processor.clean_transactions(df_loaded)
    _, pr_categorize = profile_module('app.data.processor', 'TransactionProcessor.simple_categorize', 
                                    processor, df_cleaned)
    
    # Save the profile stats
    s = io.StringIO()
    ps = pstats.Stats(pr_categorize, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    with open(os.path.join(output_dir, "profile_simple_categorize.txt"), 'w') as f:
        f.write(s.getvalue())
    
    print("Transaction processing profiling completed.")

def profile_categorization(output_dir):
    """Profile the categorization component"""
    print("\n=== Profiling Categorization ===")
    
    # Import the necessary modules
    from app.models.categorization.nlp_categorizer import TransactionCategorizer
    
    # Create a test dataset
    print("Generating test data...")
    np.random.seed(42)
    
    # Categories and sample descriptions
    categories = ['food', 'transport', 'housing', 'utilities', 'entertainment', 'shopping', 'health']
    descriptions = {
        'food': ['Grocery Store', 'Restaurant', 'Fast Food', 'Cafe', 'Food Delivery'],
        'transport': ['Gas Station', 'Uber', 'Public Transit', 'Car Payment', 'Auto Insurance'],
        'housing': ['Rent Payment', 'Mortgage', 'Home Insurance', 'Property Tax', 'HOA Fee'],
        'utilities': ['Electric Bill', 'Water Bill', 'Internet', 'Phone Bill', 'Streaming Service'],
        'entertainment': ['Movie Theater', 'Concert Tickets', 'Theme Park', 'Subscription', 'Sports Event'],
        'shopping': ['Online Purchase', 'Department Store', 'Clothing Store', 'Electronics', 'Furniture'],
        'health': ['Doctor Visit', 'Pharmacy', 'Health Insurance', 'Gym Membership', 'Dental Care']
    }
    
    # Generate training data
    train_descriptions = []
    train_categories = []
    
    for _ in range(1000):
        category = np.random.choice(categories)
        description = np.random.choice(descriptions[category])
        train_descriptions.append(description)
        train_categories.append(category)
    
    # Generate test data
    test_descriptions = []
    test_categories = []
    
    for _ in range(200):
        category = np.random.choice(categories)
        description = np.random.choice(descriptions[category])
        test_descriptions.append(description)
        test_categories.append(category)
    
    # Profile training
    print("Profiling categorizer training...")
    categorizer = TransactionCategorizer()
    _, pr_train = profile_module('app.models.categorization.nlp_categorizer', 
                               'TransactionCategorizer.train', 
                               categorizer, 
                               train_descriptions, 
                               train_categories)
    
    # Save the profile stats
    s = io.StringIO()
    ps = pstats.Stats(pr_train, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    with open(os.path.join(output_dir, "profile_categorizer_train.txt"), 'w') as f:
        f.write(s.getvalue())
    
    # Profile prediction
    print("Profiling categorizer prediction...")
    _, pr_predict = profile_module('app.models.categorization.nlp_categorizer', 
                                 'TransactionCategorizer.predict', 
                                 categorizer, 
                                 test_descriptions)
    
    # Save the profile stats
    s = io.StringIO()
    ps = pstats.Stats(pr_predict, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    with open(os.path.join(output_dir, "profile_categorizer_predict.txt"), 'w') as f:
        f.write(s.getvalue())
    
    print("Categorization profiling completed.")

def profile_forecasting(output_dir):
    """Profile the forecasting component"""
    print("\n=== Profiling Forecasting ===")
    
    # Import the necessary modules
    from app.models.forecasting.spending_forecaster import SpendingForecaster
    from app.models.time_series.time_series_processor import TimeSeriesProcessor
    
    # Create a test dataset
    print("Generating test data...")
    
    # Create time series data for multiple categories
    time_series_data = {}
    categories = ['food', 'transport', 'housing', 'utilities', 'entertainment', 'shopping', 'health']
    
    # Create date range for the past year
    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
    
    for category in categories:
        # Create synthetic time series with trend and seasonality
        np.random.seed(42 + categories.index(category))  # Different seed for each category
        
        # Base spending amount varies by category
        if category == 'housing':
            base_amount = -1500
            volatility = 100
        elif category == 'food':
            base_amount = -400
            volatility = 50
        elif category == 'transport':
            base_amount = -300
            volatility = 30
        elif category == 'utilities':
            base_amount = -200
            volatility = 20
        elif category == 'entertainment':
            base_amount = -150
            volatility = 40
        elif category == 'shopping':
            base_amount = -200
            volatility = 70
        else:  # health
            base_amount = -100
            volatility = 50
        
        # Add trend (gradual increase in spending)
        trend = np.linspace(0, -100, len(dates))
        
        # Add seasonality
        if category in ['food', 'shopping', 'entertainment']:
            # Weekly seasonality (weekends higher)
            seasonality = np.array([0, 0, 0, 0, 20, 40, 40] * (len(dates) // 7 + 1))[:len(dates)]
        elif category in ['utilities']:
            # Monthly seasonality
            monthly_pattern = np.sin(np.arange(30) * (2 * np.pi / 30)) * 30
            seasonality = np.tile(monthly_pattern, len(dates) // 30 + 1)[:len(dates)]
        elif category in ['housing']:
            # Monthly with spike at month start
            seasonality = np.zeros(len(dates))
            for i in range(0, len(dates), 30):
                if i < len(seasonality):
                    seasonality[i] = -200  # Rent/mortgage payment
        else:
            # Random pattern
            seasonality = np.zeros(len(dates))
        
        # Add noise
        noise = np.random.normal(0, volatility, len(dates))
        
        # Combine all components
        values = base_amount + trend + seasonality + noise
        
        # Create the time series DataFrame
        time_series_data[category] = pd.DataFrame({
            'date': dates,
            'amount': values
        })
    
    # Profile time series processing
    print("Profiling time series processing...")
    time_series_processor = TimeSeriesProcessor()
    
    # Create a DataFrame with all transactions
    all_transactions = []
    for category, ts_df in time_series_data.items():
        for _, row in ts_df.iterrows():
            all_transactions.append({
                'transaction_date': row['date'],
                'amount': row['amount'],
                'category': category,
                'description': f"{category.capitalize()} expense"
            })
    
    transactions_df = pd.DataFrame(all_transactions)
    
    # Profile transactions_to_time_series
    _, pr_time_series = profile_module('app.models.time_series.time_series_processor', 
                                     'TimeSeriesProcessor.transactions_to_time_series', 
                                     time_series_processor, 
                                     transactions_df,
                                     date_col='transaction_date',
                                     amount_col='amount',
                                     category_col='category')
    
    # Save the profile stats
    s = io.StringIO()
    ps = pstats.Stats(pr_time_series, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    with open(os.path.join(output_dir, "profile_time_series_processing.txt"), 'w') as f:
        f.write(s.getvalue())
    
    # Profile forecast training
    print("Profiling forecast model training...")
    forecaster = SpendingForecaster()
    
    _, pr_train_forecast = profile_module('app.models.forecasting.spending_forecaster', 
                                        'SpendingForecaster.train_models', 
                                        forecaster, 
                                        time_series_data,
                                        column='amount')
    
    # Save the profile stats
    s = io.StringIO()
    ps = pstats.Stats(pr_train_forecast, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    with open(os.path.join(output_dir, "profile_forecast_training.txt"), 'w') as f:
        f.write(s.getvalue())
    
    # Profile forecast generation
    print("Profiling forecast generation...")
    _, pr_forecast = profile_module('app.models.forecasting.spending_forecaster', 
                                  'SpendingForecaster.forecast', 
                                  forecaster, 
                                  days=30)
    
    # Save the profile stats
    s = io.StringIO()
    ps = pstats.Stats(pr_forecast, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    with open(os.path.join(output_dir, "profile_forecast_generation.txt"), 'w') as f:
        f.write(s.getvalue())
    
    print("Forecasting profiling completed.")

def profile_recommendations(output_dir):
    """Profile the recommendations component"""
    print("\n=== Profiling Recommendations ===")
    
    # Import the necessary modules
    from app.models.recommendation.recommendation_engine import RecommendationEngine
    
    # Use the same data from forecasting profiling
    print("Generating test data...")
    
    # Create a test dataset with transactions
    np.random.seed(42)
    
    # Transaction types and descriptions
    categories = ['food', 'transport', 'housing', 'utilities', 'entertainment', 'shopping', 'health']
    
    # Create date range for the past 3 months
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    
    # Generate transaction data
    transactions = []
    
    for date in dates:
        # Generate 3-6 transactions per day
        num_transactions = np.random.randint(3, 7)
        
        for _ in range(num_transactions):
            category = np.random.choice(categories)
            
            # Amount depends on category
            if category == 'housing':
                amount = -np.random.uniform(800, 2000)
            elif category == 'food':
                amount = -np.random.uniform(10, 200)
            elif category == 'transport':
                amount = -np.random.uniform(20, 150)
            elif category == 'utilities':
                amount = -np.random.uniform(50, 300)
            elif category == 'entertainment':
                amount = -np.random.uniform(10, 100)
            elif category == 'shopping':
                amount = -np.random.uniform(20, 300)
            else:  # health
                amount = -np.random.uniform(20, 500)
            
            # Add some income transactions
            if np.random.random() < 0.05:  # 5% chance
                category = 'income'
                amount = np.random.uniform(1000, 5000)
                description = 'Salary Deposit'
            else:
                description = f"{category.capitalize()} payment"
            
            transactions.append({
                'transaction_date': date,
                'amount': amount,
                'category': category,
                'description': description
            })
    
    transactions_df = pd.DataFrame(transactions)
    
    # Generate example forecasts
    forecasts = {}
    for category in categories:
        future_dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
        
        # Forecast amounts based on category
        if category == 'housing':
            base_forecast = -1500
        elif category == 'food':
            base_forecast = -500
        elif category == 'transport':
            base_forecast = -300
        elif category == 'utilities':
            base_forecast = -200
        elif category == 'entertainment':
            base_forecast = -150
        elif category == 'shopping':
            base_forecast = -200
        else:  # health
            base_forecast = -100
        
        # Add some variation
        forecast_values = base_forecast + np.random.normal(0, base_forecast * 0.1, len(future_dates))
        
        forecasts[category] = pd.DataFrame({
            'date': future_dates,
            'forecast': forecast_values,
            'lower_bound': forecast_values * 0.9,
            'upper_bound': forecast_values * 1.1
        })
    
    # Profile recommendation generation
    print("Profiling recommendation generation...")
    engine = RecommendationEngine()
    
    _, pr_recommendations = profile_module('app.models.recommendation.recommendation_engine', 
                                         'RecommendationEngine.generate_recommendations', 
                                         engine, 
                                         transactions_df,
                                         forecasts=forecasts,
                                         date_col='transaction_date',
                                         amount_col='amount',
                                         category_col='category',
                                         desc_col='description')
    
    # Save the profile stats
    s = io.StringIO()
    ps = pstats.Stats(pr_recommendations, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    with open(os.path.join(output_dir, "profile_recommendation_generation.txt"), 'w') as f:
        f.write(s.getvalue())
    
    print("Recommendations profiling completed.")

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Profile performance of Financial Health Assistant components')
    parser.add_argument('--component', '-c', choices=['transaction_processing', 'categorization', 
                                                    'forecasting', 'recommendations', 'all'],
                        default='all', help='Component to profile')
    parser.add_argument('--output', '-o', default='profiling_results',
                        help='Directory to save profiling results')
    
    args = parser.parse_args()
    
    # Run profiling
    run_profiling(args.component, args.output)
    
    print(f"\nProfiling results saved to: {args.output}")
    print("Run 'python -m pstats <profile_file>' to analyze the results interactively.")

if __name__ == "__main__":
    main()