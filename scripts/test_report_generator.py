# scripts/test_report_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.recommendation.recommendation_engine import RecommendationEngine
from app.models.recommendation.report_generator import FinancialReportGenerator
from app.models.forecasting.spending_forecaster import SpendingForecaster

def test_report_generator():
    """Test the financial report generator with transaction data and recommendations"""
    
    print("=" * 50)
    print("Testing Financial Report Generator")
    print("=" * 50)
    
    # Check if we have real data
    transactions_path = 'app/data/processed/transactions_clean.csv'
    recommendations_path = 'app/data/processed/recommendations.csv'
    
    # Load transaction data
    if os.path.exists(transactions_path):
        print(f"Using real transaction data from {transactions_path}")
        df = pd.read_csv(transactions_path)
    else:
        print("No real transaction data found. Creating synthetic data for testing...")
        from scripts.test_recommendations import create_synthetic_data
        df = create_synthetic_data()
    
    print(f"Loaded {len(df)} transactions")
    
    # Ensure columns are properly formatted
    date_col = 'transaction_date'
    amount_col = 'amount'
    
    # Convert date column to datetime if needed
    if date_col in df.columns and not pd.api.types.is_datetime64_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Check for amount column or create it
    if amount_col not in df.columns:
        if 'withdrawal' in df.columns and 'deposit' in df.columns:
            # Convert withdrawal to negative, deposit to positive
            df['withdrawal'] = pd.to_numeric(df['withdrawal'], errors='coerce').fillna(0)
            df['deposit'] = pd.to_numeric(df['deposit'], errors='coerce').fillna(0)
            df[amount_col] = df['deposit'] - df['withdrawal']
            print("Created 'amount' column from withdrawal and deposit columns")
    
    # Identify column names
    print("Available columns:", df.columns.tolist())
    
    # Determine category and description columns
    category_col = 'category' if 'category' in df.columns else None
    desc_col = 'description' if 'description' in df.columns else ('TRANSACTION DETAILS' if 'TRANSACTION DETAILS' in df.columns else None)
    
    # Load recommendations if available, otherwise generate them
    recommendations = []
    if os.path.exists(recommendations_path):
        print(f"Loading existing recommendations from {recommendations_path}")
        recommendations_df = pd.read_csv(recommendations_path)
        
        # Convert DataFrame to list of dictionaries
        recommendations = recommendations_df.to_dict('records')
        print(f"Loaded {len(recommendations)} recommendations")
    else:
        print("Generating new recommendations...")
        # Initialize recommendation engine
        engine = RecommendationEngine(threshold_percentile=75)
        
        # Generate recommendations
        recommendations = engine.generate_recommendations(
            df,
            date_col=date_col,
            amount_col=amount_col,
            category_col=category_col,
            desc_col=desc_col,
            limit=10
        )
        print(f"Generated {len(recommendations)} recommendations")
    
    # Check for forecasting models if available
    forecasts = None
    forecasting_model_dir = 'app/models/forecasting/saved_models'
    
    if os.path.exists(forecasting_model_dir) and os.listdir(forecasting_model_dir):
        try:
            print("\nLoading forecasting models...")
            # Load forecaster and forecasts
            forecaster = SpendingForecaster()
            forecaster.load_models(forecasting_model_dir)
            forecasts = forecaster.forecast(days=30)
            print(f"Loaded forecasts for {len(forecasts)} categories")
        except Exception as e:
            print(f"Error loading forecasts: {str(e)}")
            print("Proceeding without forecasting models")
    
    # Create sample user info
    user_info = {
        'user_id': 'test_user_123',
        'name': 'Test User',
        'email': 'testuser@example.com',
        'report_date': datetime.now().isoformat()
    }
    
    # Initialize report generator
    report_generator = FinancialReportGenerator()
    
    # Create output directory
    report_dir = 'app/data/processed/reports'
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate financial report
    try:
        report = report_generator.generate_report(
            transactions_df=df,
            recommendations=recommendations,
            forecasts=forecasts,
            user_info=user_info,
            output_dir=report_dir,
            date_col=date_col,
            amount_col=amount_col,
            category_col=category_col
        )
        
        print(f"\nFinancial report generated successfully!")
        
        # Display report sections
        print("\nReport Sections:")
        for section, data in report['sections'].items():
            print(f"- {section.replace('_', ' ').title()}")
        
        # Display some key metrics from the report
        if 'summary' in report['sections']:
            summary = report['sections']['summary']
            print("\nKey Metrics:")
            print(f"- Date Range: {summary['date_range']['start_date']} to {summary['date_range']['end_date']}")
            print(f"- Total Income: ${summary['total']['income']:.2f}")
            print(f"- Total Expenses: ${summary['total']['expenses']:.2f}")
            print(f"- Net Cash Flow: ${summary['total']['net_cashflow']:.2f}")
            print(f"- Savings Rate: {summary['savings_rate']:.1f}%")
        
        # List visualizations created
        print("\nVisualizations:")
        for section_name, section_data in report['sections'].items():
            if 'visualizations' in section_data:
                print(f"- {section_name.replace('_', ' ').title()}:")
                for viz_name, viz_path in section_data['visualizations'].items():
                    print(f"  - {viz_name.replace('_', ' ').title()}: {viz_path}")
        
        return True
    except Exception as e:
        print(f"Error generating financial report: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('app/data/processed', exist_ok=True)
    os.makedirs('app/data/processed/reports', exist_ok=True)
    
    # Test report generator
    success = test_report_generator()
    
    if success:
        print("\nReport generator tested successfully!")
    else:
        print("\nReport generator test failed")