"""
Integration tests for the recommendation engine
"""
import os
import sys
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.models.recommendation.recommendation_engine import RecommendationEngine
from app.models.forecasting.spending_forecaster import SpendingForecaster

@pytest.fixture
def sample_transactions():
    """Create a sample transaction dataset for testing recommendations"""
    # Create 100 transactions spread over the last 3 months
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    np.random.seed(42)  # For reproducibility
    
    # Generate realistic transaction data
    categories = ['food', 'transport', 'housing', 'utilities', 'entertainment', 'shopping', 'health']
    category_weights = [0.25, 0.15, 0.30, 0.10, 0.08, 0.07, 0.05]  # Realistic distribution
    
    data = {
        'transaction_date': dates,
        'amount': [],
        'category': [],
        'description': []
    }
    
    # Common transaction descriptions by category
    descriptions = {
        'food': ['Grocery Store', 'Restaurant', 'Fast Food', 'Coffee Shop', 'Food Delivery'],
        'transport': ['Gas Station', 'Uber', 'Public Transit', 'Car Payment', 'Auto Insurance'],
        'housing': ['Rent Payment', 'Mortgage', 'Home Insurance', 'Property Tax', 'HOA Fee'],
        'utilities': ['Electric Bill', 'Water Bill', 'Internet', 'Phone Bill', 'Streaming Service'],
        'entertainment': ['Movie Theater', 'Concert Tickets', 'Theme Park', 'Subscription', 'Sports Event'],
        'shopping': ['Online Purchase', 'Department Store', 'Clothing Store', 'Electronics', 'Furniture'],
        'health': ['Doctor Visit', 'Pharmacy', 'Health Insurance', 'Gym Membership', 'Dental Care']
    }
    
    # Generate amounts and categories
    for _ in range(100):
        category = np.random.choice(categories, p=category_weights)
        data['category'].append(category)
        
        # Set realistic amount ranges by category
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
        elif category == 'health':
            amount = -np.random.uniform(20, 500)
        else:
            amount = -np.random.uniform(10, 100)
            
        data['amount'].append(amount)
        
        # Select a random description for the category
        desc = np.random.choice(descriptions[category])
        data['description'].append(desc)
    
    # Add some income transactions
    income_dates = pd.date_range(end=datetime.now(), periods=3, freq='30D')
    for date in income_dates:
        # Use reset_index to avoid pandas concatenation errors
        data['transaction_date'] = pd.concat([data['transaction_date'], pd.Series([date])]).reset_index(drop=True)
        data['amount'].append(np.random.uniform(2000, 5000))  # Income
        data['category'].append('income')
        data['description'].append('Salary Deposit')
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_forecasts():
    """Create sample forecasts for recommendation testing"""
    np.random.seed(42)
    categories = ['food', 'transport', 'housing', 'utilities', 'entertainment', 'shopping', 'health']
    
    forecast_days = 30
    dates = pd.date_range(start=datetime.now(), periods=forecast_days, freq='D')
    
    forecasts = {}
    
    for category in categories:
        # Base amount varies by category
        if category == 'housing':
            base = 1500
            variance = 100
        elif category == 'food':
            base = 500
            variance = 50
        elif category == 'transport':
            base = 300
            variance = 30
        elif category == 'utilities':
            base = 200
            variance = 20
        elif category == 'entertainment':
            base = 150
            variance = 40
        elif category == 'shopping':
            base = 200
            variance = 70
        else:  # health
            base = 100
            variance = 50
            
        # Create the forecast with some randomness
        forecast_values = base + np.random.normal(0, variance, size=forecast_days)
        
        # Create DataFrame
        forecasts[category] = pd.DataFrame({
            'date': dates,
            'forecast': -forecast_values  # Negative for expenses
        })
        
        # Add confidence intervals
        forecasts[category]['lower_bound'] = forecasts[category]['forecast'] * 0.9
        forecasts[category]['upper_bound'] = forecasts[category]['forecast'] * 1.1
    
    return forecasts

def test_recommendation_engine_integration(sample_transactions, sample_forecasts, tmp_path):
    """Test the recommendation engine end-to-end"""
    # Initialize the recommendation engine
    engine = RecommendationEngine(threshold_percentile=80)
    
    # Generate recommendations
    recommendations = engine.generate_recommendations(
        sample_transactions,
        forecasts=sample_forecasts,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category',
        desc_col='description',
        limit=10
    )
    
    # Verify recommendation structure
    assert isinstance(recommendations, list), "Recommendations should be a list"
    assert len(recommendations) > 0, "Should generate at least one recommendation"
    assert len(recommendations) <= 10, "Should respect the limit parameter"
    
    # Check recommendation fields
    required_fields = ['message', 'type', 'priority', 'savings_potential', 'category', 'confidence']
    for rec in recommendations:
        for field in required_fields:
            assert field in rec, f"Recommendation should contain '{field}' field"
        
        # Check values have correct types
        assert isinstance(rec['message'], str), "Message should be a string"
        assert isinstance(rec['type'], str), "Type should be a string"
        assert isinstance(rec['priority'], (int, float)), "Priority should be numeric"
        assert isinstance(rec['savings_potential'], (int, float)), "Savings potential should be numeric"
        assert rec['savings_potential'] >= 0, "Savings potential should be non-negative"
        assert 0 <= rec['confidence'] <= 1, "Confidence should be between 0 and 1"
    
    # Test persistence - save to CSV
    output_csv = tmp_path / "recommendations.csv"
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv(output_csv, index=False)
    assert output_csv.exists(), "Recommendations CSV should be created"
    
    # Reload and verify
    loaded_df = pd.read_csv(output_csv)
    assert len(loaded_df) == len(recommendations), "Loaded recommendations should match original count"
    
    # Test visualization
    viz_path = tmp_path / "recommendation_viz.png"
    try:
        engine.visualize_recommendations(recommendations, str(viz_path))
        assert viz_path.exists(), "Visualization file should be created"
    except Exception as e:
        pytest.fail(f"Visualization failed: {str(e)}")
    
    # Test recommendation generation with different parameters
    limited_recommendations = engine.generate_recommendations(
        sample_transactions,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category',
        desc_col='description',
        limit=3
    )
    assert len(limited_recommendations) <= 3, "Should respect the smaller limit parameter"
    
    # Test with different threshold
    strict_engine = RecommendationEngine(threshold_percentile=95)
    strict_recommendations = strict_engine.generate_recommendations(
        sample_transactions,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category',
        desc_col='description'
    )
    
    # We expect fewer recommendations with stricter threshold
    assert len(strict_recommendations) <= len(recommendations), \
        "Stricter threshold should produce fewer or equal recommendations"
    
    print("Integration test of recommendation engine completed successfully!")