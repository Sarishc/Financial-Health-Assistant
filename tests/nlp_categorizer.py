# test_nlp_categorizer.py
import pandas as pd
from app.models.categorization.nlp_categorizer import TransactionCategorizer
import os

def test_categorization_model():
    """Test the trained NLP categorization model"""
    
    # Check if model exists
    model_path = 'app/models/categorization/transaction_categorizer.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return False
    
    # Load categorizer
    categorizer = TransactionCategorizer(model_path)
    
    # Test on some examples
    test_descriptions = [
        "AMAZON PURCHASE",
        "UBER EATS FOOD DELIVERY",
        "SUPERMARKET GROCERIES",
        "MONTHLY RENT PAYMENT",
        "SALARY DEPOSIT",
        "NETFLIX SUBSCRIPTION",
        "GAS STATION FUEL",
        "ATM WITHDRAWAL",
        "DOCTOR APPOINTMENT PAYMENT",
        "ELECTRIC BILL PAYMENT"
    ]
    
    # Make predictions
    predictions = categorizer.predict(test_descriptions)
    
    # Print results
    print("\nCategorization Results:")
    print("-" * 50)
    for desc, pred in zip(test_descriptions, predictions):
        print(f"Description: '{desc}'")
        print(f"Predicted category: '{pred}'")
        print("-" * 50)
    
    # Load some real transaction data for testing
    transactions_path = 'app/data/processed/transactions_clean.csv'
    if os.path.exists(transactions_path):
        df = pd.read_csv(transactions_path)
        
        # Map columns if needed
        if 'TRANSACTION DETAILS' in df.columns and 'description' not in df.columns:
            df['description'] = df['TRANSACTION DETAILS']
        
        # Select a sample of transactions
        sample = df.sample(min(10, len(df))) if len(df) > 0 else df
        
        if 'description' in sample.columns:
            # Predict categories
            sample_descriptions = sample['description'].fillna("").tolist()
            sample_predictions = categorizer.predict(sample_descriptions)
            
            # Print results
            print("\nReal Transaction Samples:")
            print("-" * 50)
            for desc, pred in zip(sample_descriptions, sample_predictions):
                print(f"Description: '{desc}'")
                print(f"Predicted category: '{pred}'")
                print("-" * 50)
    
    return True

if __name__ == "__main__":
    test_categorization_model()