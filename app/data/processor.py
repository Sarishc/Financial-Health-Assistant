import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict, Any, Optional

class TransactionProcessor:
    """Class for processing and transforming transaction data"""
    
    def __init__(self):
        """Initialize the transaction processor"""
        self.category_keywords = {
            'food': ['grocery', 'restaurant', 'cafe', 'food', 'dining'],
            'transport': ['uber', 'lyft', 'gas', 'fuel', 'transit', 'train', 'bus'],
            'shopping': ['amazon', 'walmart', 'target', 'purchase', 'store'],
            'utilities': ['electric', 'water', 'gas', 'utility', 'bill', 'phone', 'internet'],
            'entertainment': ['movie', 'netflix', 'spotify', 'hulu', 'game'],
            'health': ['doctor', 'pharmacy', 'medical', 'fitness', 'gym'],
            'housing': ['rent', 'mortgage', 'home'],
            'income': ['salary', 'deposit', 'payment received'],
            'other': []
        }
    
    def load_transactions(self, filepath: str) -> pd.DataFrame:
        """
        Load transaction data from a file
        
        Args:
            filepath: Path to the transaction data file
            
        Returns:
            DataFrame containing transaction data
        """
        if str(filepath).endswith('.csv'):
            df = pd.read_csv(filepath)
        elif str(filepath).endswith('.xlsx') or str(filepath).endswith('.xls'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        return df
    
    def clean_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize transaction data
        
        Args:
            df: DataFrame containing raw transaction data
            
        Returns:
            Cleaned DataFrame with standardized column names and formats
        """
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Standardize column names (will be expanded)
        column_mapping = {
            # Map common column names to our standard names
            'DATE': 'transaction_date',
            'date': 'transaction_date',
            'transaction date': 'transaction_date',
            'trans_date': 'transaction_date',
            'TRANSACTION DETAILS': 'description',
            'desc': 'description',
            'memo': 'description',
            'transaction': 'description',
            'transactiondescription': 'description',
            'amt': 'amount',
            'debit': 'amount',
            'credit': 'amount',
            'transaction_amount': 'amount',
            'categ': 'category',
            'transaction_category': 'category'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in cleaned_df.columns:
                cleaned_df[new_col] = cleaned_df[old_col]
        
        # Create amount column if it doesn't exist
        if 'amount' not in cleaned_df.columns:
            if ' WITHDRAWAL AMT ' in cleaned_df.columns and ' DEPOSIT AMT ' in cleaned_df.columns:
                # Convert to numeric
                cleaned_df[' WITHDRAWAL AMT '] = pd.to_numeric(cleaned_df[' WITHDRAWAL AMT '], errors='coerce').fillna(0)
                cleaned_df[' DEPOSIT AMT '] = pd.to_numeric(cleaned_df[' DEPOSIT AMT '], errors='coerce').fillna(0)
                # Create amount column (deposits positive, withdrawals negative)
                cleaned_df['amount'] = cleaned_df[' DEPOSIT AMT '] - cleaned_df[' WITHDRAWAL AMT ']
        
        # Convert date to datetime
        if 'transaction_date' in cleaned_df.columns:
            cleaned_df['transaction_date'] = pd.to_datetime(cleaned_df['transaction_date'], errors='coerce')
        
        # Handle missing descriptions
        if 'description' in cleaned_df.columns:
            cleaned_df['description'] = cleaned_df['description'].fillna('Unknown')
        
        # Convert amount to float if not already
        if 'amount' in cleaned_df.columns and not pd.api.types.is_numeric_dtype(cleaned_df['amount']):
            # Try to clean amount string (remove currency symbols, etc)
            cleaned_df['amount'] = cleaned_df['amount'].astype(str).str.replace('[$,]', '', regex=True)
            cleaned_df['amount'] = pd.to_numeric(cleaned_df['amount'], errors='coerce')
        
        # Ensure we have a category column
        if 'category' not in cleaned_df.columns:
            cleaned_df['category'] = None
        
        # Ensure required columns exist
        required_columns = ['transaction_date', 'description', 'amount']
        missing_columns = [col for col in required_columns if col not in cleaned_df.columns]
        if missing_columns:
            raise ValueError(f"Required columns missing: {missing_columns}")
        
        # Drop rows with critical missing values
        for col in required_columns:
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df.dropna(subset=[col])
        
        return cleaned_df
    
    def simple_categorize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform simple rule-based categorization of transactions
        
        Args:
            df: DataFrame containing transaction data with 'description' column
            
        Returns:
            DataFrame with added or updated 'category' column
        """
        # Make a copy to avoid modifying the original
        categorized_df = df.copy()
        
        # Only categorize uncategorized transactions
        mask = categorized_df['category'].isna() | (categorized_df['category'] == '')
        
        def assign_category(description: str) -> str:
            """Assign a category based on keywords in the description"""
            if not isinstance(description, str):
                return 'other'
                
            description = description.lower()
            
            for category, keywords in self.category_keywords.items():
                if any(keyword in description for keyword in keywords):
                    return category
            
            return 'other'
        
        # Apply categorization function
        categorized_df.loc[mask, 'category'] = categorized_df.loc[mask, 'description'].apply(assign_category)
        
        return categorized_df
    
    def process_transactions(self, filepath: str) -> pd.DataFrame:
        """
        Complete pipeline to load, clean, and categorize transactions
        
        Args:
            filepath: Path to the transaction data file
            
        Returns:
            Processed DataFrame ready for analysis
        """
        # Load data
        df = self.load_transactions(filepath)
        
        # Clean data
        df = self.clean_transactions(df)
        
        # Categorize data
        df = self.simple_categorize(df)
        
        return df