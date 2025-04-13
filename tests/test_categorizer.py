# Create a file named test_categorizer.py in the project root
import pandas as pd
from app.models.categorization.nlp_categorizer import TransactionCategorizer

# Load our processed data
df = pd.read_csv('app/data/processed/transactions_clean.csv')
print(f"Loaded {len(df)} transactions")

# Check if we have the necessary columns
if 'description' not in df.columns or 'category' not in df.columns:
    print("Error: Missing required columns (description or category)")
    exit(1)

# Drop rows with missing values
df = df.dropna(subset=['description', 'category'])
print(f"Using {len(df)} transactions with valid descriptions and categories")

# Create and train the categorizer
categorizer = TransactionCategorizer()
accuracy = categorizer.train(df['description'].tolist(), df['category'].tolist())

print(f"Model training complete. Accuracy: {accuracy:.4f}")

# Test on some examples
test_descriptions = [
    "AMAZON PURCHASE",
    "Uber ride",
    "Grocery store purchase",
    "Monthly rent payment",
    "ATM withdrawal"
]

# Make predictions
if len(df) > 10:  # Only predict if we had enough data to train
    predictions = categorizer.predict(test_descriptions)
    for desc, pred in zip(test_descriptions, predictions):
        print(f"Description: '{desc}' → Predicted category: '{pred}'")
    
    # Save the model
    categorizer.save_model('app/models/categorization/transaction_categorizer.joblib')