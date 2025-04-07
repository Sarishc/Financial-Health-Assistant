# Notebook: financial_data_exploration.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import os

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# -------------------------------------------------------
# 1. Kaggle Bank Marketing Dataset
# -------------------------------------------------------
# Dataset link: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset
# Download and place in app/data/raw/bank_marketing.csv

def analyze_bank_marketing_dataset():
    # Load dataset
    file_path = 'app/data/raw/bank_marketing.csv'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print("Please download the Bank Marketing dataset from Kaggle and place it in the correct location.")
        return
    
    df = pd.read_csv(file_path, sep=';')
    
    # Display basic information
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nColumn Information:")
    print(df.info())
    
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Explore categorical features
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        print(f"\n{col} distribution:")
        counts = df[col].value_counts()
        print(counts)
        
        # Plot distribution
        plt.figure(figsize=(10, 6))
        counts.head(10).plot(kind='bar')
        plt.title(f'Top 10 {col} Categories')
        plt.ylabel('Count')
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(f'notebooks/bank_marketing_{col}.png')
    
    # Explore numeric features
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        print(f"\n{col} statistics:")
        print(df[col].describe())
        
        # Plot distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'notebooks/bank_marketing_{col}_dist.png')
    
    # Transform into transaction-like data for the project (adapted for financial health assistant)
    print("\nTransforming bank marketing data into transaction-like format...")
    
    # Create synthetic transaction data from customer information
    transactions = []
    
    # Get unique customer IDs (just use index for demonstration)
    customer_ids = df.index.tolist()
    
    # Create date range for the last 6 months
    end_date = datetime.now()
    start_date = datetime(end_date.year, end_date.month - 6, 1) if end_date.month > 6 else datetime(end_date.year - 1, end_date.month + 6, 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Define transaction types based on job and marital status
    transaction_types = {
        'admin': ['Office Supplies', 'Business Lunch', 'Software Subscription'],
        'blue-collar': ['Hardware Store', 'Auto Repair', 'Work Boots'],
        'entrepreneur': ['Business Investment', 'Consulting Fee', 'Office Rent'],
        'housemaid': ['Cleaning Supplies', 'Home Services', 'Utilities'],
        'management': ['Business Dinner', 'Travel Expense', 'Conference Fee'],
        'retired': ['Pharmacy', 'Grocery Store', 'Medical Visit'],
        'self-employed': ['Business Equipment', 'Professional Services', 'Marketing'],
        'services': ['Tools', 'Uniform', 'Professional License'],
        'student': ['Textbooks', 'School Supplies', 'Online Course'],
        'technician': ['Technical Parts', 'Professional Tools', 'Training Course'],
        'unemployed': ['Grocery Store', 'Job Application', 'Public Transport'],
        'unknown': ['Miscellaneous', 'General Purchase', 'Retail Shopping']
    }
    
    # Categories mapping
    category_mapping = {
        'Office Supplies': 'shopping',
        'Business Lunch': 'food',
        'Software Subscription': 'utilities',
        'Hardware Store': 'shopping',
        'Auto Repair': 'transport',
        'Work Boots': 'shopping',
        'Business Investment': 'other',
        'Consulting Fee': 'income',
        'Office Rent': 'housing',
        'Cleaning Supplies': 'shopping',
        'Home Services': 'housing',
        'Utilities': 'utilities',
        'Business Dinner': 'food',
        'Travel Expense': 'transport',
        'Conference Fee': 'other',
        'Pharmacy': 'health',
        'Grocery Store': 'food',
        'Medical Visit': 'health',
        'Business Equipment': 'shopping',
        'Professional Services': 'other',
        'Marketing': 'other',
        'Tools': 'shopping',
        'Uniform': 'shopping',
        'Professional License': 'other',
        'Textbooks': 'shopping',
        'School Supplies': 'shopping',
        'Online Course': 'entertainment',
        'Technical Parts': 'shopping',
        'Professional Tools': 'shopping',
        'Training Course': 'other',
        'Job Application': 'other',
        'Public Transport': 'transport',
        'Miscellaneous': 'other',
        'General Purchase': 'shopping',
        'Retail Shopping': 'shopping',
        'Salary Deposit': 'income',
        'Interest Payment': 'income',
        'ATM Withdrawal': 'other'
    }
    
    # Generate transactions for each customer
    np.random.seed(42)  # For reproducibility
    
    for idx, row in df.iterrows():
        # Get customer details
        job = row['job'] if row['job'] in transaction_types else 'unknown'
        balance = row['balance']
        
        # Determine number of transactions (more for higher balance)
        num_transactions = max(3, int(np.log(abs(balance) + 1) / 2)) if balance > 0 else 3
        
        # Generate transactions
        for _ in range(num_transactions):
            # Generate transaction date
            trans_date = np.random.choice(date_range)
            
            # Generate transaction description
            if balance > 1000 and np.random.random() < 0.2:
                description = np.random.choice(['Salary Deposit', 'Interest Payment'])
                amount = np.random.uniform(500, 3000)
            elif balance < 0 and np.random.random() < 0.3:
                description = 'ATM Withdrawal'
                amount = -np.random.uniform(50, 300)
            else:
                description = np.random.choice(transaction_types[job])
                # Expense (negative amount)
                amount = -np.random.uniform(10, 500)
            
            # Get category
            category = category_mapping[description]
            
            # Add transaction
            transactions.append({
                'transaction_date': trans_date,
                'description': description,
                'amount': amount,
                'category': category,
                'customer_id': idx
            })
    
    # Create transactions dataframe
    transactions_df = pd.DataFrame(transactions)
    
    # Sort by date
    transactions_df = transactions_df.sort_values('transaction_date')
    
    # Save transactions to CSV
    output_path = 'app/data/processed/bank_marketing_transactions.csv'
    transactions_df.to_csv(output_path, index=False)
    print(f"Generated {len(transactions_df)} synthetic transactions from bank marketing data")
    print(f"Saved to {output_path}")
    
    # Analyze generated transactions
    print("\nGenerated Transactions Sample:")
    print(transactions_df.head())
    
    # Analyze categories
    print("\nTransaction Categories:")
    category_counts = transactions_df['category'].value_counts()
    print(category_counts)
    
    # Plot category distribution
    plt.figure(figsize=(12, 6))
    category_counts.plot(kind='bar')
    plt.title('Transaction Categories from Bank Marketing Data')
    plt.ylabel('Count')
    plt.xlabel('Category')
    plt.tight_layout()
    plt.savefig('notebooks/bank_marketing_categories.png')
    
    # Monthly transaction analysis
    transactions_df['month'] = transactions_df['transaction_date'].dt.to_period('M')
    monthly_spending = transactions_df.groupby('month')['amount'].sum()
    
    plt.figure(figsize=(12, 6))
    monthly_spending.plot(kind='bar')
    plt.title('Monthly Net Cash Flow')
    plt.ylabel('Amount')
    plt.xlabel('Month')
    plt.tight_layout()
    plt.savefig('notebooks/bank_marketing_monthly.png')
    
    return transactions_df

# -------------------------------------------------------
# 2. Bank Transaction Classification Dataset
# -------------------------------------------------------
# Dataset link: https://www.kaggle.com/datasets/apoorvwatsky/bank-transaction-data
# Download and place in app/data/raw/bank_transactions_classification.csv

def analyze_bank_transactions():
    # Load dataset
    file_path = 'app/data/raw/bank_transactions_classification.csv'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print("Please download the dataset from Kaggle and place it in the correct location.")
        return
    
    df = pd.read_csv(file_path)
    
    # Display basic information
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nColumn Information:")
    print(df.info())
    
    # Explore transaction descriptions
    if 'Description' in df.columns:
        print("\nSample Descriptions:")
        print(df['Description'].sample(10))
        
        # Count word frequencies in descriptions
        all_words = ' '.join(df['Description'].str.lower()).split()
        word_counts = pd.Series(all_words).value_counts()
        
        print("\nMost common words in descriptions:")
        print(word_counts.head(20))
        
        # Plot word frequencies
        plt.figure(figsize=(12, 6))
        word_counts.head(15).plot(kind='bar')
        plt.title('Most Common Words in Transaction Descriptions')
        plt.ylabel('Count')
        plt.xlabel('Word')
        plt.tight_layout()
        plt.savefig('notebooks/bank_word_freq.png')
    
    # Explore categories if available
    if 'Category' in df.columns:
        print("\nTransaction Categories:")
        category_counts = df['Category'].value_counts()
        print(category_counts)
        
        # Plot category distribution
        plt.figure(figsize=(12, 6))
        category_counts.plot(kind='bar')
        plt.title('Transaction Categories')
        plt.ylabel('Count')
        plt.xlabel('Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('notebooks/bank_categories.png')
    
    return df

# -------------------------------------------------------
# 3. Create Synthetic Financial Transaction Dataset
# -------------------------------------------------------
# If you don't have access to the Kaggle datasets, this function will create synthetic data

def create_synthetic_transactions(n_transactions=1000):
    """Create synthetic transaction data for testing"""
    
    # Categories and their associated merchants
    categories = {
        'food': ['McDonalds', 'Chipotle', 'Whole Foods', 'Trader Joes', 'Starbucks', 'Subway', 'KFC'],
        'transport': ['Uber', 'Lyft', 'Shell', 'Exxon', 'BP', 'AMTrak', 'Delta Airlines'],
        'shopping': ['Amazon', 'Target', 'Walmart', 'Best Buy', 'Home Depot', 'Macys', 'Nike'],
        'utilities': ['AT&T', 'Verizon', 'Electricity Co', 'Water Services', 'Internet Provider', 'Gas Company'],
        'entertainment': ['Netflix', 'Spotify', 'Movie Theater', 'Hulu', 'Disney+', 'Xbox', 'Steam'],
        'health': ['CVS Pharmacy', 'Walgreens', 'Hospital Co-pay', 'Gym Membership', 'Dental Care'],
        'housing': ['Rent Payment', 'Mortgage', 'Home Insurance', 'Maintenance Fee', 'Furniture Store'],
        'income': ['Salary', 'Freelance Payment', 'Tax Refund', 'Interest', 'Dividend'],
    }
    
    # Create merchants list with their categories
    merchants = []
    merchant_categories = {}
    for category, merchant_list in categories.items():
        for merchant in merchant_list:
            merchants.append(merchant)
            merchant_categories[merchant] = category
    
    # Generate random dates in the last 12 months
    end_date = datetime.now()
    start_date = datetime(end_date.year - 1, end_date.month, 1)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_transactions)
    
    # Generate random transactions
    data = {
        'transaction_date': dates,
        'description': np.random.choice(merchants, size=n_transactions),
        'amount': np.zeros(n_transactions),
        'category': np.empty(n_transactions, dtype=object)
    }
    
    # Set amounts based on categories and add category
    for i, merchant in enumerate(data['description']):
        category = merchant_categories[merchant]
        data['category'][i] = category
        
        # Different amount ranges for different categories
        if category == 'food':
            data['amount'][i] = -np.random.uniform(5, 100)  # Negative for spending
        elif category == 'transport':
            data['amount'][i] = -np.random.uniform(10, 200)
        elif category == 'shopping':
            data['amount'][i] = -np.random.uniform(20, 500)
        elif category == 'utilities':
            data['amount'][i] = -np.random.uniform(50, 300)
        elif category == 'entertainment':
            data['amount'][i] = -np.random.uniform(10, 150)
        elif category == 'health':
            data['amount'][i] = -np.random.uniform(20, 1000)
        elif category == 'housing':
            data['amount'][i] = -np.random.uniform(500, 3000)
        elif category == 'income':
            data['amount'][i] = np.random.uniform(1000, 10000)  # Positive for income
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = 'app/data/raw/synthetic_transactions.csv'
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset created and saved to {output_path}")
    
    # Analyze the synthetic data
    print("\nSynthetic Dataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Explore categories
    print("\nTransaction Categories:")
    category_counts = df['category'].value_counts()
    print(category_counts)
    
    # Plot category distribution
    plt.figure(figsize=(10, 6))
    category_counts.plot(kind='bar')
    plt.title('Synthetic Data: Transaction Categories')
    plt.ylabel('Count')
    plt.xlabel('Category')
    plt.tight_layout()
    plt.savefig('notebooks/synthetic_categories.png')
    
    # Plot spending by category
    spending_by_category = df.groupby('category')['amount'].sum().sort_values()
    plt.figure(figsize=(10, 6))
    spending_by_category.plot(kind='barh')
    plt.title('Synthetic Data: Total Amount by Category')
    plt.xlabel('Total Amount')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig('notebooks/synthetic_spending.png')
    
    # Monthly spending
    df['month'] = df['transaction_date'].dt.to_period('M')
    monthly_spending = df.groupby('month')['amount'].sum()
    
    plt.figure(figsize=(12, 6))
    monthly_spending.plot()
    plt.title('Synthetic Data: Monthly Net Cash Flow')
    plt.ylabel('Amount')
    plt.xlabel('Month')
    plt.tight_layout()
    plt.savefig('notebooks/synthetic_monthly.png')
    
    return df

# -------------------------------------------------------
# 4. Combined Analysis Function
# -------------------------------------------------------

def analyze_all_datasets():
    """Analyze all available datasets and provide summary"""
    datasets = {}
    summary = {}
    
    # Try to load and analyze all datasets
    try:
        print("=" * 50)
        print("Analyzing Bank Marketing Dataset")
        print("=" * 50)
        datasets['bank_marketing'] = analyze_bank_marketing_dataset()
        if datasets['bank_marketing'] is not None:
            summary['bank_marketing'] = {
                'rows': datasets['bank_marketing'].shape[0],
                'columns': datasets['bank_marketing'].shape[1],
                'has_categories': 'category' in datasets['bank_marketing'].columns,
                'date_column': 'transaction_date' if 'transaction_date' in datasets['bank_marketing'].columns else None
            }
    except Exception as e:
        print(f"Error analyzing Bank Marketing dataset: {str(e)}")
    
    try:
        print("\n" + "=" * 50)
        print("Analyzing Bank Transactions Dataset")
        print("=" * 50)
        datasets['bank'] = analyze_bank_transactions()
        if datasets['bank'] is not None:
            summary['bank'] = {
                'rows': datasets['bank'].shape[0],
                'columns': datasets['bank'].shape[1],
                'has_categories': 'Category' in datasets['bank'].columns,
                'has_descriptions': 'Description' in datasets['bank'].columns
            }
    except Exception as e:
        print(f"Error analyzing Bank dataset: {str(e)}")
    
    # Always create synthetic data as a fallback
    print("\n" + "=" * 50)
    print("Creating Synthetic Transaction Dataset")
    print("=" * 50)
    datasets['synthetic'] = create_synthetic_transactions(n_transactions=1500)
    summary['synthetic'] = {
        'rows': datasets['synthetic'].shape[0],
        'columns': datasets['synthetic'].shape[1],
        'has_categories': True,
        'date_column': 'transaction_date'
    }
    
    # Print summary
    print("\n" + "=" * 50)
    print("Dataset Summary")
    print("=" * 50)
    for name, stats in summary.items():
        print(f"\n{name.capitalize()} Dataset:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
    
    # Recommendation
    print("\n" + "=" * 50)
    print("Recommendation")
    print("=" * 50)
    
    if 'bank_marketing' in datasets and datasets['bank_marketing'] is not None:
        print("Primary recommendation: Use the transformed Bank Marketing dataset as your primary dataset.")
        recommended = 'bank_marketing'
    elif 'bank' in datasets and datasets['bank'] is not None:
        print("Primary recommendation: Use the Bank Transactions dataset as your primary dataset.")
        recommended = 'bank'
    else:
        print("Primary recommendation: Use the Synthetic dataset as your starting point.")
        recommended = 'synthetic'
    
    print("\nThe selected dataset has been saved and is ready for further processing.")
    return datasets[recommended]

# -------------------------------------------------------
# Main execution
# -------------------------------------------------------

if __name__ == "__main__":
    # Make sure directories exist
    os.makedirs('app/data/raw', exist_ok=True)
    os.makedirs('app/data/processed', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    
    # Run the analysis
    final_dataset = analyze_all_datasets()
    
    # Save the selected dataset to processed folder
    processed_path = 'app/data/processed/transactions_clean.csv'
    final_dataset.to_csv(processed_path, index=False)
    print(f"\nFinal dataset saved to {processed_path}")
    
    print("\nDay 1 data analysis complete!")