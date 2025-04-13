# train_categorizer.py
import pandas as pd
from app.models.categorization.nlp_categorizer import TransactionCategorizer
import os

def train_categorization_model():
    """Train and save the NLP categorization model"""
    
    # Load transaction data
    transactions_path = 'app/data/processed/transactions_clean.csv'
    
    if not os.path.exists(transactions_path):
        print(f"Error: {transactions_path} not found")
        return False
    
    df = pd.read_csv(transactions_path)
    print(f"Loaded {len(df)} transactions")
    
    # Check columns
    print("Available columns:", df.columns.tolist())
    
    # Map columns if needed
    column_mapping = {
        'TRANSACTION DETAILS': 'description',
    }
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Check if we have category column
    if 'category' not in df.columns:
        print("No 'category' column found in the data. Adding basic categorization...")
        
        # Add basic categorization
        def simple_categorize(description):
            if not isinstance(description, str):
                return 'other'
            
            description = str(description).lower()
            
            categories = {
                'food': ['grocery', 'restaurant', 'cafe', 'food', 'dining', 'eat'],
                'transport': ['uber', 'lyft', 'gas', 'fuel', 'transit', 'train', 'travel'],
                'shopping': ['amazon', 'walmart', 'target', 'purchase', 'store', 'buy'],
                'utilities': ['electric', 'water', 'gas', 'utility', 'bill', 'phone'],
                'entertainment': ['movie', 'netflix', 'spotify', 'hulu', 'game'],
                'health': ['doctor', 'pharmacy', 'medical', 'fitness', 'hospital'],
                'income': ['salary', 'deposit', 'payment received', 'credit'],
                'transfer': ['transfer', 'neft', 'imps', 'rtgs', 'withdrawal'],
                'other': []
            }
            
            for category, keywords in categories.items():
                if any(keyword in description for keyword in keywords):
                    return category
            
            return 'other'
        
        if 'description' in df.columns:
            df['category'] = df['description'].apply(simple_categorize)
        elif 'TRANSACTION DETAILS' in df.columns:
            df['category'] = df['TRANSACTION DETAILS'].apply(simple_categorize)
        else:
            print("Error: No description column found for categorization")
            return False
        
        # Save categorized data back to CSV
        df.to_csv(transactions_path, index=False)
        print(f"Added categories and saved back to {transactions_path}")
    
    # Initialize categorizer
    categorizer = TransactionCategorizer()
    
    # Prepare training data
    try:
        desc_col = 'description' if 'description' in df.columns else 'TRANSACTION DETAILS'
        descriptions, categories = categorizer.prepare_training_data(df, 
                                                                   description_col=desc_col, 
                                                                   category_col='category')
        print(f"Prepared {len(descriptions)} training examples")
    except Exception as e:
        print(f"Error preparing training data: {str(e)}")
        return False
    
    # Train the model
    try:
        accuracy = categorizer.train(descriptions, categories)
        print(f"Model trained with accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False
    
    # Save the model
    model_path = 'app/models/categorization/transaction_categorizer.joblib'
    try:
        categorizer.save_model(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('app/models/categorization', exist_ok=True)
    
    # Train the model
    success = train_categorization_model()
    
    if success:
        print("Categorization model training completed successfully")
    else:
        print("Categorization model training failed")