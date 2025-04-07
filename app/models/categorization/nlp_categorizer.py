import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

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
        
        # Load pre-trained model if specified
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.is_trained = True
                print(f"Loaded categorization model from {model_path}")
            except Exception as e:
                print(f"Failed to load model from {model_path}: {str(e)}")
    
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
        
        return self.model.predict(descriptions)
    
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