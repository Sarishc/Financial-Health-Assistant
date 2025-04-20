# app/api/routes/categories.py
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from typing import List, Optional, Dict, Any
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Import schemas
from app.api.schemas.category import Category, CategoryStats, CategoryList, CategoryHierarchy
from app.api.schemas.messages import Message
from app.api.auth.auth import get_current_active_user
from app.api.schemas.auth import User

# Add parent directory to path for importing app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import categorizer if available
try:
    from app.models.categorization.nlp_categorizer import TransactionCategorizer
    CATEGORIZER_AVAILABLE = True
except ImportError:
    CATEGORIZER_AVAILABLE = False

# Create router
router = APIRouter()

# Path to data files
TRANSACTIONS_PATH = "app/data/processed/transactions_clean.csv"
CATEGORIES_PATH = "app/data/processed/categories.csv"
CATEGORIZER_PATH = "app/models/categorization/transaction_categorizer.joblib"

# Default categories
DEFAULT_CATEGORIES = [
    {"name": "food", "description": "Food and dining expenses", "color": "#FF5733", "icon": "utensils"},
    {"name": "transport", "description": "Transportation expenses", "color": "#33FF57", "icon": "car"},
    {"name": "utilities", "description": "Utility bills", "color": "#3357FF", "icon": "bolt"},
    {"name": "entertainment", "description": "Entertainment expenses", "color": "#FF33A8", "icon": "film"},
    {"name": "shopping", "description": "Shopping expenses", "color": "#A833FF", "icon": "shopping-bag"},
    {"name": "health", "description": "Healthcare expenses", "color": "#33FFF6", "icon": "heartbeat"},
    {"name": "housing", "description": "Housing expenses", "color": "#FFC733", "icon": "home"},
    {"name": "income", "description": "Income sources", "color": "#57FF33", "icon": "dollar-sign"},
    {"name": "other", "description": "Other expenses", "color": "#999999", "icon": "question-circle"}
]

# Helper functions
def load_categories():
    """Load categories from CSV file or create default if not exists"""
    if not os.path.exists(CATEGORIES_PATH):
        # Create default categories
        df = pd.DataFrame(DEFAULT_CATEGORIES)
        df.to_csv(CATEGORIES_PATH, index=False)
        return df
    
    return pd.read_csv(CATEGORIES_PATH)

def save_categories(df):
    """Save categories to CSV file"""
    df.to_csv(CATEGORIES_PATH, index=False)

def load_transactions():
    """Load transactions from CSV file"""
    if not os.path.exists(TRANSACTIONS_PATH):
        return pd.DataFrame()
    
    df = pd.read_csv(TRANSACTIONS_PATH)
    
    # Convert date columns to datetime
    date_columns = ['transaction_date', 'DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df

# Category routes
@router.get("/categories", response_model=CategoryList)
async def get_categories(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a list of all categories
    """
    df = load_categories()
    
    # Convert to Pydantic models
    categories = []
    for _, row in df.iterrows():
        category = Category(
            name=row['name'],
            description=row['description'] if 'description' in row and pd.notna(row['description']) else None,
            color=row['color'] if 'color' in row and pd.notna(row['color']) else None,
            icon=row['icon'] if 'icon' in row and pd.notna(row['icon']) else None,
            parent_category=row['parent_category'] if 'parent_category' in row and pd.notna(row['parent_category']) else None
        )
        categories.append(category)
    
    return {"categories": categories, "total": len(categories)}

@router.post("/categories", response_model=Category)
async def create_category(
    category: Category,
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new category
    
    - **name**: Category name (must be unique)
    - **description**: Category description (optional)
    - **color**: Hex color code for display (optional)
    - **icon**: Icon name for display (optional)
    - **parent_category**: Parent category name for hierarchical organization (optional)
    """
    df = load_categories()
    
    # Check if category already exists
    if category.name in df['name'].values:
        raise HTTPException(status_code=400, detail="Category already exists")
    
    # If parent category is specified, check if it exists
    if category.parent_category and category.parent_category not in df['name'].values:
        raise HTTPException(status_code=400, detail="Parent category does not exist")
    
    # Create new category
    new_category = {
        'name': category.name,
        'description': category.description,
        'color': category.color,
        'icon': category.icon,
        'parent_category': category.parent_category
    }
    
    # Add to DataFrame
    df = pd.concat([df, pd.DataFrame([new_category])], ignore_index=True)
    
    # Save to file
    save_categories(df)
    
    return category

@router.get("/categories/{category_name}", response_model=Category)
async def get_category(
    category_name: str = Path(..., description="The name of the category to get"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a single category by name
    
    - **category_name**: Name of the category to retrieve
    """
    df = load_categories()
    
    # Find category
    category = df[df['name'] == category_name]
    
    if category.empty:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Convert to Pydantic model
    row = category.iloc[0]
    return Category(
        name=row['name'],
        description=row['description'] if 'description' in row and pd.notna(row['description']) else None,
        color=row['color'] if 'color' in row and pd.notna(row['color']) else None,
        icon=row['icon'] if 'icon' in row and pd.notna(row['icon']) else None,
        parent_category=row['parent_category'] if 'parent_category' in row and pd.notna(row['parent_category']) else None
    )

@router.put("/categories/{category_name}", response_model=Category)
async def update_category(
    category_name: str = Path(..., description="The name of the category to update"),
    category: Category = Body(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update a category
    
    - **category_name**: Name of the category to update
    - **category**: Updated category data
    """
    df = load_categories()
    
    # Find category
    if category_name not in df['name'].values:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # If name is being changed, ensure new name doesn't already exist
    if category.name != category_name and category.name in df['name'].values:
        raise HTTPException(status_code=400, detail="Category with new name already exists")
    
    # If parent category is specified, check if it exists
    if category.parent_category and category.parent_category not in df['name'].values:
        raise HTTPException(status_code=400, detail="Parent category does not exist")
    
    # Update category
    mask = df['name'] == category_name
    df.loc[mask, 'name'] = category.name
    df.loc[mask, 'description'] = category.description
    df.loc[mask, 'color'] = category.color
    df.loc[mask, 'icon'] = category.icon
    df.loc[mask, 'parent_category'] = category.parent_category
    
    # Save changes
    save_categories(df)
    
    return category

@router.delete("/categories/{category_name}", response_model=Message)
async def delete_category(
    category_name: str = Path(..., description="The name of the category to delete"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete a category
    
    - **category_name**: Name of the category to delete
    """
    df = load_categories()
    
    # Find category
    if category_name not in df['name'].values:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Prevent deletion of certain built-in categories
    if category_name in ['income', 'other']:
        raise HTTPException(status_code=400, detail="Cannot delete built-in category")
    
    # Delete category
    df = df[df['name'] != category_name]
    
    # Save changes
    save_categories(df)
    
    return {"message": "Category deleted successfully"}

@router.get("/categories/{category_name}/stats", response_model=CategoryStats)
async def get_category_stats(
    category_name: str = Path(..., description="The name of the category to get stats for"),
    start_date: Optional[datetime] = Query(None, description="Start date for statistics"),
    end_date: Optional[datetime] = Query(None, description="End date for statistics"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get statistics for a specific category
    
    - **category_name**: Name of the category to get stats for
    - **start_date**: Start date for calculating statistics (default: 30 days ago)
    - **end_date**: End date for calculating statistics (default: today)
    """
    # Check if category exists
    category_df = load_categories()
    if category_name not in category_df['name'].values:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Load transactions
    df = load_transactions()
    
    # Ensure we have category column
    category_col = 'category'
    if category_col not in df.columns:
        raise HTTPException(status_code=400, detail="Transaction data does not include categories")
    
    # Set default date range if not provided
    if end_date is None:
        end_date = datetime.now()
    
    if start_date is None:
        start_date = end_date - timedelta(days=30)
    
    # Determine date column
    date_col = 'transaction_date' if 'transaction_date' in df.columns else 'DATE'
    
    # Filter by date range
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    filtered_df = df[mask]
    
    # Filter by category
    category_transactions = filtered_df[filtered_df[category_col] == category_name]
    
    if category_transactions.empty:
        # Return zero stats if no transactions
        return {
            "name": category_name,
            "transaction_count": 0,
            "total_amount": 0.0,
            "average_transaction": 0.0,
            "percentage_of_total": 0.0,
            "month_to_month_change": None
        }
    
    # Determine amount column
    amount_col = 'amount'
    if amount_col not in filtered_df.columns:
        if 'withdrawal' in filtered_df.columns and 'deposit' in filtered_df.columns:
            filtered_df[amount_col] = filtered_df['deposit'] - filtered_df['withdrawal']
        elif ' WITHDRAWAL AMT ' in filtered_df.columns and ' DEPOSIT AMT ' in filtered_df.columns:
            filtered_df[amount_col] = filtered_df[' DEPOSIT AMT '] - filtered_df[' WITHDRAWAL AMT ']
    
    # Calculate statistics
    transaction_count = len(category_transactions)
    
    # Handle the case where amount_col isn't in category_transactions yet
    if amount_col not in category_transactions.columns:
        if 'withdrawal' in category_transactions.columns and 'deposit' in category_transactions.columns:
            category_transactions[amount_col] = category_transactions['deposit'] - category_transactions['withdrawal']
        elif ' WITHDRAWAL AMT ' in category_transactions.columns and ' DEPOSIT AMT ' in category_transactions.columns:
            category_transactions[amount_col] = category_transactions[' DEPOSIT AMT '] - category_transactions[' WITHDRAWAL AMT ']
    
    total_amount = abs(category_transactions[amount_col].sum())
    average_transaction = abs(category_transactions[amount_col].mean())
    
    # Calculate percentage of total
    total_transactions_amount = abs(filtered_df[amount_col].sum())
    percentage_of_total = (total_amount / total_transactions_amount * 100) if total_transactions_amount > 0 else 0
    
    # Calculate month-to-month change
    month_to_month_change = None
    
    # Group by month and calculate the change
    if date_col in category_transactions.columns:
        category_transactions['month'] = category_transactions[date_col].dt.to_period('M')
        monthly_amounts = category_transactions.groupby('month')[amount_col].sum().abs()
        
        if len(monthly_amounts) >= 2:
            current_month = monthly_amounts.iloc[-1]
            previous_month = monthly_amounts.iloc[-2]
            
            if previous_month > 0:
                month_to_month_change = ((current_month - previous_month) / previous_month) * 100
    
    return {
        "name": category_name,
        "transaction_count": transaction_count,
        "total_amount": float(total_amount),
        "average_transaction": float(average_transaction),
        "percentage_of_total": float(percentage_of_total),
        "month_to_month_change": float(month_to_month_change) if month_to_month_change is not None else None
    }

@router.get("/categories/hierarchy", response_model=List[CategoryHierarchy])
async def get_category_hierarchy(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get hierarchical structure of categories
    """
    df = load_categories()
    
    # Add parent_category column if it doesn't exist
    if 'parent_category' not in df.columns:
        df['parent_category'] = None
    
    # Build hierarchy
    hierarchy = []
    
    # Get root categories (no parent)
    root_categories = df[df['parent_category'].isna() | (df['parent_category'] == '')]['name'].tolist()
    
    # Build tree for each root category
    for root in root_categories:
        node = build_category_tree(df, root)
        hierarchy.append(node)
    
    return hierarchy

def build_category_tree(df, category_name):
    """Recursively build a category hierarchy tree"""
    # Get child categories
    children = df[df['parent_category'] == category_name]['name'].tolist()
    
    # Create node
    node = CategoryHierarchy(name=category_name, subcategories=[])
    
    # Add children
    for child in children:
        child_node = build_category_tree(df, child)
        node.subcategories.append(child_node)
    
    return node

@router.post("/categories/categorize-transactions", response_model=Message)
async def categorize_transactions(
    current_user: User = Depends(get_current_active_user)
):
    """
    Categorize uncategorized transactions using the trained model
    """
    if not CATEGORIZER_AVAILABLE:
        raise HTTPException(status_code=400, detail="Categorization module not available")
    
    if not os.path.exists(CATEGORIZER_PATH):
        raise HTTPException(status_code=400, detail="No trained categorizer model found")
    
    # Load transactions
    df = load_transactions()
    
    # Ensure we have description column
    desc_col = 'description'
    if desc_col not in df.columns:
        if 'TRANSACTION DETAILS' in df.columns:
            df[desc_col] = df['TRANSACTION DETAILS']
        else:
            raise HTTPException(status_code=400, detail="Transaction data does not include descriptions")
    
    # Filter uncategorized transactions
    if 'category' in df.columns:
        uncategorized = df['category'].isna() | (df['category'] == '')
    else:
        df['category'] = None
        uncategorized = pd.Series([True] * len(df))
    
    # Skip if no uncategorized transactions
    if not any(uncategorized):
        return {"message": "No uncategorized transactions found"}
    
    # Get descriptions to categorize
    descriptions = df.loc[uncategorized, desc_col].tolist()
    
    # Initialize categorizer
    categorizer = TransactionCategorizer(model_path=CATEGORIZER_PATH)
    
    # Predict categories
    try:
        predicted_categories = categorizer.predict(descriptions)
        
        # Update categories
        df.loc[uncategorized, 'category'] = predicted_categories
        
        # Save updated transactions
        df.to_csv(TRANSACTIONS_PATH, index=False)
        
        return {"message": f"Successfully categorized {len(predicted_categories)} transactions"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error categorizing transactions: {str(e)}")