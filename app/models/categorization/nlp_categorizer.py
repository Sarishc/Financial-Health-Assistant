import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
import re
import string

class TransactionCategorizer:
    """
    NLP-based transaction categorization model
    """
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the categorizer
        
        Args:
            model_path: Path to load a pre-trained model (optional)
        """
        # Default model pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB())
        ])
        
        self.is_trained = False
        self.categories = []
        
        # Standard categories
        self.standard_categories = [
            'food', 'transport', 'shopping', 'utilities', 
            'entertainment', 'health', 'housing', 'income', 
            'transfer', 'education', 'travel', 'other'
        ]
        
        # Load pre-trained model if specified
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.is_trained = True
                print(f"Loaded categorization model from {model_path}")
            except Exception as e:
                print(f"Failed to load model from {model_path}: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess transaction description for better NLP performance
        
        Args:
            text: Raw transaction description
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numeric-only tokens
        text = re.sub(r'\b\d+\b', 'NUM', text)
        
        return text
    
    def prepare_training_data(self, df: pd.DataFrame, 
                              description_col: str = 'description', 
                              category_col: str = 'category') -> tuple:
        """
        Prepare training data from transaction DataFrame
        
        Args:
            df: DataFrame containing transactions
            description_col: Column name containing transaction descriptions
            category_col: Column name containing transaction categories
            
        Returns:
            Tuple of (descriptions, categories)
        """
        if description_col not in df.columns or category_col not in df.columns:
            raise ValueError(f"Missing required columns. Need {description_col} and {category_col}")
        
        # Drop rows with missing values in required columns
        df = df.dropna(subset=[description_col, category_col])
        
        # Get unique categories
        self.categories = sorted(df[category_col].unique())
        
        # Preprocess descriptions
        descriptions = [self.preprocess_text(desc) for desc in df[description_col]]
        categories = df[category_col].tolist()
        
        return descriptions, categories
    
    def train(self, descriptions: List[str], categories: List[str]) -> float:
        """
        Train the categorization model
        
        Args:
            descriptions: List of transaction descriptions
            categories: List of corresponding categories
            
        Returns:
            Accuracy score on validation set
        """
        if len(descriptions) != len(categories):
            raise ValueError("Length of descriptions and categories must match")
        
        if len(descriptions) < 10:
            raise ValueError("Need at least 10 examples to train a model")
        
        # Store unique categories
        self.categories = sorted(set(categories))
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            descriptions, categories, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        accuracy = self.model.score(X_val, y_val)
        
        self.is_trained = True
        print(f"Trained categorization model with {len(X_train)} examples. Validation accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def predict(self, descriptions: List[str]) -> List[str]:
        """
        Predict categories for transaction descriptions
        
        Args:
            descriptions: List of transaction descriptions
            
        Returns:
            List of predicted categories
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet")
        
        # Preprocess descriptions
        processed_descriptions = [self.preprocess_text(desc) for desc in descriptions]
        
        return self.model.predict(processed_descriptions)
    
    def save_model(self, output_path: str) -> None:
        """
        Save the trained model to disk
        
        Args:
            output_path: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self.model, output_path)
        print(f"Saved categorization model to {output_path}")
    
    def categorize_dataframe(self, df: pd.DataFrame, 
                             description_col: str = 'description',
                             category_col: str = 'category') -> pd.DataFrame:
        """
        Categorize transactions in a DataFrame
        
        Args:
            df: DataFrame containing transactions
            description_col: Column name containing transaction descriptions
            category_col: Column name to store predicted categories
            
        Returns:
            DataFrame with predicted categories
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet")
            
        if description_col not in df.columns:
            raise ValueError(f"Missing required column: {description_col}")
            
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Get descriptions
        descriptions = result_df[description_col].fillna("").tolist()
        
        # Predict categories
        predicted_categories = self.predict(descriptions)
        
        # Add predictions to DataFrame
        result_df[category_col] = predicted_categories
        
        return result_df