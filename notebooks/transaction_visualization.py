import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Create output directory for visualizations
os.makedirs('notebooks/visualizations', exist_ok=True)

# Load the processed transaction data
df = pd.read_csv('app/data/processed/transactions_clean.csv')
print(f"Loaded {len(df)} transactions")
print("Available columns:", df.columns.tolist())

# Map columns to standard names
column_mapping = {
    'DATE': 'transaction_date',
    'TRANSACTION DETAILS': 'description',
    ' WITHDRAWAL AMT ': 'withdrawal',
    ' DEPOSIT AMT ': 'deposit',
    'BALANCE AMT': 'balance'
}

# Rename columns if they exist
for old_col, new_col in column_mapping.items():
    if old_col in df.columns:
        df.rename(columns={old_col: new_col}, inplace=True)

# Create an amount column (negative for withdrawals, positive for deposits)
if 'withdrawal' in df.columns and 'deposit' in df.columns:
    # Convert to numeric, replacing non-numeric values with 0
    df['withdrawal'] = pd.to_numeric(df['withdrawal'], errors='coerce').fillna(0)
    df['deposit'] = pd.to_numeric(df['deposit'], errors='coerce').fillna(0)
    
    # Create amount column (negative for withdrawals, positive for deposits)
    df['amount'] = df['deposit'] - df['withdrawal']
    print("Created 'amount' column from withdrawal and deposit columns")

# Simplified categorization function without complex lookups
def simple_categorize(df):
    # Define categories based on keywords
    category_keywords = {
        'food': ['grocery', 'restaurant', 'cafe', 'food', 'dining', 'eat'],
        'transport': ['uber', 'lyft', 'gas', 'fuel', 'transit', 'train'],
        'shopping': ['amazon', 'walmart', 'target', 'purchase', 'store'],
        'utilities': ['electric', 'water', 'gas', 'utility', 'bill', 'phone'],
        'entertainment': ['movie', 'netflix', 'spotify', 'hulu', 'game'],
        'health': ['doctor', 'pharmacy', 'medical', 'fitness', 'gym'],
        'income': ['salary', 'deposit', 'credit'],
        'other': []
    }
    
    # Add category column
    categories = []
    
    for idx, row in df.iterrows():
        if 'description' in df.columns:
            description = str(row['description']).lower() if pd.notna(row['description']) else ''
            
            matched = False
            for category, keywords in category_keywords.items():
                if any(keyword in description for keyword in keywords):
                    categories.append(category)
                    matched = True
                    break
                    
            if not matched:
                categories.append('other')
        else:
            categories.append('other')
    
    df['category'] = categories
    return df

# Add categorization
if 'description' in df.columns:
    print("Adding categorization based on transaction details...")
    df = simple_categorize(df)
    print(f"Categorized transactions into {df['category'].nunique()} categories")
else:
    # If no description column, create a default category
    df['category'] = 'uncategorized'
    print("No description column found. Using default category.")

# 1. Category Distribution
plt.figure(figsize=(12, 6))
category_counts = df['category'].value_counts()
category_counts.plot(kind='bar')
plt.title('Transaction Categories Distribution')
plt.ylabel('Count')
plt.xlabel('Category')
plt.tight_layout()
plt.savefig('notebooks/visualizations/category_distribution.png')
plt.close()
print("Created category distribution visualization")

# 2. Transaction Amounts by Category (using withdrawals for expenses)
plt.figure(figsize=(12, 6))
if 'withdrawal' in df.columns:
    category_amounts = df.groupby('category')['withdrawal'].sum().sort_values(ascending=False)
    category_amounts = category_amounts[category_amounts > 0]  # Only show categories with withdrawals
elif 'amount' in df.columns:
    df_expenses = df[df['amount'] < 0].copy()
    df_expenses['amount'] = df_expenses['amount'].abs()
    category_amounts = df_expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
else:
    category_amounts = pd.Series()
    print("No amount column available for category spending visualization")

if not category_amounts.empty:
    category_amounts.plot(kind='bar')
    plt.title('Total Spending by Category')
    plt.ylabel('Total Amount')
    plt.xlabel('Category')
    plt.tight_layout()
    plt.savefig('notebooks/visualizations/category_amounts.png')
    plt.close()
    print("Created category amounts visualization")

# 3. Time Series Analysis
if 'transaction_date' in df.columns:
    try:
        # Convert to datetime if not already
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        
        # Filter out rows with invalid dates
        df_dates = df.dropna(subset=['transaction_date'])
        
        # Only proceed if valid dates exist
        if not df_dates.empty:
            # Group by month
            df_dates['month'] = df_dates['transaction_date'].dt.to_period('M')
            
            # Monthly withdrawals (expenses)
            if 'withdrawal' in df_dates.columns:
                monthly_spending = df_dates.groupby('month')['withdrawal'].sum()
                
                plt.figure(figsize=(14, 6))
                monthly_spending.plot()
                plt.title('Monthly Withdrawals Over Time')
                plt.ylabel('Total Withdrawal Amount')
                plt.xlabel('Month')
                plt.tight_layout()
                plt.savefig('notebooks/visualizations/monthly_spending.png')
                plt.close()
                print("Created monthly withdrawal visualization")
            
            # Monthly spending by category
            if 'withdrawal' in df_dates.columns and df_dates['withdrawal'].sum() > 0:
                try:
                    # Use pivot table with withdrawal amount
                    pivot_df = df_dates.pivot_table(
                        index='month', 
                        columns='category', 
                        values='withdrawal', 
                        aggfunc='sum'
                    ).fillna(0)
                    
                    if not pivot_df.empty and pivot_df.size > 0:
                        plt.figure(figsize=(14, 8))
                        pivot_df.plot(kind='area', stacked=True)
                        plt.title('Monthly Spending by Category')
                        plt.ylabel('Amount')
                        plt.xlabel('Month')
                        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.tight_layout()
                        plt.savefig('notebooks/visualizations/monthly_category_spending.png')
                        plt.close()
                        print("Created monthly category spending visualization")
                except Exception as e:
                    print(f"Error creating monthly category spending visualization: {str(e)}")
        else:
            print("No valid transaction dates found for time series analysis")
    except Exception as e:
        print(f"Error in time series analysis: {str(e)}")
else:
    print("No transaction_date column available for time series analysis")

print("\nVisualizations created in notebooks/visualizations/")

# Display some basic statistics
print("\nTransaction Statistics:")
print(f"Total number of transactions: {len(df)}")

if 'transaction_date' in df.columns:
    valid_dates = pd.to_datetime(df['transaction_date'], errors='coerce').dropna()
    if not valid_dates.empty:
        print(f"Date range: {valid_dates.min()} to {valid_dates.max()}")
    else:
        print("No valid dates found")
        
if 'withdrawal' in df.columns and 'deposit' in df.columns:
    print(f"Total withdrawals (spending): {df['withdrawal'].sum():.2f}")
    print(f"Total deposits (income): {df['deposit'].sum():.2f}")
    print(f"Net cash flow: {df['deposit'].sum() - df['withdrawal'].sum():.2f}")
elif 'amount' in df.columns:
    expenses = df[df['amount'] < 0]['amount'].sum()
    income = df[df['amount'] > 0]['amount'].sum()
    print(f"Total spending: ${abs(expenses):.2f}")
    print(f"Total income: ${income:.2f}")
    print(f"Net cash flow: ${df['amount'].sum():.2f}")

# Print info for README update
top_categories = category_counts.head(3).index.tolist()
largest_expense_category = category_amounts.head(1).index.tolist()[0] if not category_amounts.empty and len(category_amounts) > 0 else 'N/A'

print("\nFor README Update:")
print(f"- Successfully processed transaction data with {len(df)} transactions")
print(f"- Identified {len(category_counts)} different spending categories")
print(f"- Top spending categories: {', '.join(top_categories)}")
print(f"- Largest expense category: {largest_expense_category}")