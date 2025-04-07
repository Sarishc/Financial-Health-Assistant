#!/bin/bash
# setup_project.sh - Script to set up the Financial Health Assistant project structure

# Exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Financial Health Assistant project...${NC}"

# Create main project directory if it doesn't exist
if [ ! -d "financial-health-assistant" ]; then
    mkdir -p financial-health-assistant
fi

cd financial-health-assistant

# Create Python virtual environment
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
python -m venv venv

# Activate the virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix/MacOS
    source venv/bin/activate
fi

# Create project structure
echo -e "${YELLOW}Creating project directory structure...${NC}"

# Main application directory
mkdir -p app/api/routes
mkdir -p app/api/schemas
mkdir -p app/data/raw
mkdir -p app/data/processed
mkdir -p app/models/categorization
mkdir -p app/models/forecasting
mkdir -p app/models/recommendation
mkdir -p app/utils
mkdir -p app/frontend

# Other directories
mkdir -p notebooks
mkdir -p tests
mkdir -p docs
mkdir -p scripts

# Create initial files
echo -e "${YELLOW}Creating initial files...${NC}"

# Create main app files
cat > app/__init__.py << EOL
# Financial Health Assistant
# Main package initialization
EOL

cat > app/main.py << EOL
"""
Main application entry point for the Financial Health Assistant
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="Financial Health Assistant API",
    description="API for analyzing financial transactions and providing recommendations",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import API routes
# This will be uncommented as routes are implemented
# from app.api.routes import transactions, categories, forecasts, recommendations

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Financial Health Assistant API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOL

# Create API structure
cat > app/api/__init__.py << EOL
# API package initialization
EOL

cat > app/api/routes/__init__.py << EOL
# API routes initialization
EOL

# Create data utilities
cat > app/data/__init__.py << EOL
# Data module initialization
EOL

cat > app/data/processor.py << EOL
"""
Transaction data processing module
"""
import pandas as pd
from datetime import datetime
import re
from typing import List, Dict, Any, Optional

class TransactionProcessor:
    """Class for processing and transforming transaction data"""
    
    def __init__(self):
        """Initialize the transaction processor"""
        # This will be expanded as the project progresses
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
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
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
            'date': 'transaction_date',
            'transaction date': 'transaction_date',
            'trans_date': 'transaction_date',
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
                cleaned_df.rename(columns={old_col: new_col}, inplace=True)
        
        # Ensure required columns exist
        required_columns = ['transaction_date', 'description', 'amount']
        for col in required_columns:
            if col not in cleaned_df.columns:
                raise ValueError(f"Required column '{col}' not found in transaction data")
        
        # Convert date to datetime
        cleaned_df['transaction_date'] = pd.to_datetime(cleaned_df['transaction_date'], errors='coerce')
        
        # Handle missing descriptions
        cleaned_df['description'] = cleaned_df['description'].fillna('Unknown')
        
        # Convert amount to float if not already
        if not pd.api.types.is_numeric_dtype(cleaned_df['amount']):
            # Try to clean amount string (remove currency symbols, etc)
            cleaned_df['amount'] = cleaned_df['amount'].astype(str).str.replace('[$,]', '', regex=True)
            cleaned_df['amount'] = pd.to_numeric(cleaned_df['amount'], errors='coerce')
        
        # Ensure we have a category column
        if 'category' not in cleaned_df.columns:
            cleaned_df['category'] = None
        
        # Drop rows with critical missing values
        cleaned_df = cleaned_df.dropna(subset=['transaction_date', 'amount'])
        
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
            description = description.lower()
            
            for category, keywords in self.category_keywords.items():
                if any(keyword in description for keyword in keywords):
                    return category
            
            return 'other'
        
        # Apply categorization function
        categorized_df.loc[mask, 'category'] = categorized_df.loc[mask, 'description'].apply(assign_category)
        
        return categorized_df

# Create utils files
EOL

cat > app/utils/__init__.py << EOL
# Utilities module initialization
EOL

cat > app/utils/config.py << EOL
"""
Configuration utilities for the Financial Health Assistant
"""
import os
from pathlib import Path
from pydantic import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    """Application settings"""
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Financial Health Assistant"
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./financial_health.db")
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-for-development-only")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8000"]
    
    # File paths
    DATA_DIR: str = "app/data"
    RAW_DATA_DIR: str = "app/data/raw"
    PROCESSED_DATA_DIR: str = "app/data/processed"
    MODEL_DIR: str = "app/models"
    
    class Config:
        case_sensitive = True

# Create settings instance
settings = Settings()
EOL

cat > app/utils/logger.py << EOL
"""
Logging utilities for the Financial Health Assistant
"""
import logging
import sys
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)

# Create a default logger for the application
logger = get_logger("financial_health_assistant")
EOL

# Create model files
cat > app/models/__init__.py << EOL
# Models package initialization
EOL

cat > app/models/categorization/__init__.py << EOL
# Categorization models initialization
EOL

cat > app/models/categorization/nlp_categorizer.py << EOL
"""
NLP-based transaction categorization model
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

from app.utils.logger import logger

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
                logger.info(f"Loaded categorization model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {str(e)}")
    
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
        logger.info(f"Trained categorization model with {len(X_train)} examples. Validation accuracy: {accuracy:.4f}")
        
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
        logger.info(f"Saved categorization model to {output_path}")
EOL

# Create Docker files
cat > Dockerfile << EOL
FROM python:3.9-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOL

cat > docker-compose.yml << EOL
version: '3.8'

services:
  web:
    build: .
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    depends_on:
      - db
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/financial_health
      - SECRET_KEY=development_secret_key

  db:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data/
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=financial_health
    ports:
      - "5432:5432"

volumes:
  postgres_data:
EOL

# Create .env file for development
cat > .env << EOL
# Development environment variables
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/financial_health
SECRET_KEY=development_secret_key_change_in_production
DEBUG=True
EOL

# Create .gitignore
cat > .gitignore << EOL
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# Virtual environments
venv/
env/
ENV/

# IDE settings
.idea/
.vscode/
*.swp
*.swo

# Environment variables
.env
.env.*

# Data files
*.csv
*.xls
*.xlsx
*.sqlite3
app/data/raw/
app/data/processed/

# Logs
logs/
*.log

# Model files
*.pkl
*.joblib
*.h5

# Mac OS
.DS_Store
EOL

# Create a README.md file
cat > README.md << EOL
# Financial Health Assistant

An ML-powered system that analyzes financial transactions, predicts future spending patterns, and provides personalized recommendations for saving money.

## Features

- Transaction data processing and categorization using NLP
- Time-series analysis and forecasting of spending patterns
- Personalized saving recommendations
- Web-based dashboard for financial insights

## Getting Started

### Prerequisites

- Python 3.9+
- PostgreSQL (optional, SQLite works for development)
- Docker (optional)

### Installation

1. Clone the repository
   \`\`\`bash
   git clone https://github.com/yourusername/financial-health-assistant.git
   cd financial-health-assistant
   \`\`\`

2. Set up a virtual environment
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \`\`\`

3. Install dependencies
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. Set up environment variables
   \`\`\`bash
   cp .env.example .env
   # Edit .env file with your configuration
   \`\`\`

5. Run the application
   \`\`\`bash
   uvicorn app.main:app --reload
   \`\`\`

### Using Docker

Alternatively, you can use Docker to run the application:

\`\`\`bash
docker-compose up -d
\`\`\`

## Project Structure

- \`app/\`: Main application
  - \`api/\`: API routes and schemas
  - \`data/\`: Data processing utilities
  - \`models/\`: ML models for categorization, forecasting, etc.
  - \`utils/\`: Utility functions
  - \`frontend/\`: Frontend assets (to be added)
- \`notebooks/\`: Jupyter notebooks for data exploration
- \`tests/\`: Unit and integration tests
- \`docs/\`: Documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
EOL

# Create a basic requirements.txt file
cat > requirements.txt << EOL
# API
fastapi>=0.68.0
uvicorn[standard]>=0.15.0
pydantic>=1.8.0
python-multipart>=0.0.5
python-dotenv>=0.19.0

# Database
sqlalchemy>=1.4.23
psycopg2-binary>=2.9.1
alembic>=1.7.1

# Data processing
pandas>=1.3.2
numpy>=1.21.2
openpyxl>=3.0.9

# Machine learning
scikit-learn>=0.24.2
nltk>=3.6.3
spacy>=3.1.2
prophet>=1.0.1
tensorflow>=2.6.0
statsmodels>=0.13.0

# Visualization
matplotlib>=3.4.3
seaborn>=0.11.2

# Testing
pytest>=6.2.5
pytest-cov>=2.12.1

# Development
jupyter>=1.0.0
black>=21.8b0
flake8>=3.9.2
isort>=5.9.3
EOL

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Initialize git repository
echo -e "${YELLOW}Initializing git repository...${NC}"
git init
git add .
git commit -m "Initial project setup"

echo -e "${GREEN}Project setup complete!${NC}"
echo -e "To activate the virtual environment, run:"
echo -e "  ${YELLOW}source venv/bin/activate${NC}  # On Windows: ${YELLOW}venv\\Scripts\\activate${NC}"
echo -e "To start the development server, run:"
echo -e "  ${YELLOW}uvicorn app.main:app --reload${NC}"