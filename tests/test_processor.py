# Create a file named test_processor.py in the project root
import pandas as pd
from app.data.processor import TransactionProcessor

# Test the processor with our clean data
processor = TransactionProcessor()
df = processor.load_transactions('app/data/processed/transactions_clean.csv')
print(f"Loaded {len(df)} transactions")

# Process the data
processed_df = processor.process_transactions('app/data/processed/transactions_clean.csv')
print(f"Processed {len(processed_df)} transactions")

# Show category distribution
print("\nCategory distribution:")
print(processed_df['category'].value_counts())