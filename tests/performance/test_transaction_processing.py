"""
Performance tests for transaction processing component
"""
import os
import sys
import pandas as pd
import numpy as np
import pytest
import time
from datetime import datetime, timedelta
import cProfile
import pstats
import io
import matplotlib.pyplot as plt

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.data.processor import TransactionProcessor
from app.utils.performance import optimize_dataframe, parallelize_dataframe

@pytest.fixture
def large_transaction_dataset(size=100000):
    """Generate a large dataset of transactions for performance testing"""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(end=datetime.now(), periods=size, freq='H')
    
    # Transaction types and common descriptions
    transaction_types = {
        'food': ['GROCERY STORE', 'RESTAURANT', 'FAST FOOD', 'CAFE'],
        'transport': ['GAS STATION', 'UBER', 'LYFT', 'CAR PAYMENT'],
        'housing': ['RENT', 'MORTGAGE', 'HOME INSURANCE'],
        'utilities': ['ELECTRIC BILL', 'WATER BILL', 'INTERNET', 'PHONE'],
        'entertainment': ['MOVIE', 'NETFLIX', 'SPOTIFY', 'GAMING'],
        'shopping': ['AMAZON', 'WALMART', 'TARGET', 'MACY\'S'],
        'health': ['DOCTOR', 'PHARMACY', 'HEALTH INSURANCE'],
        'income': ['SALARY', 'DIRECT DEPOSIT', 'PAYMENT RECEIVED']
    }
    
    # Generate transaction descriptions and categories
    all_categories = []
    all_descriptions = []
    
    for _ in range(size):
        # 10% income, 90% expenses
        if np.random.random() < 0.1:
            category = 'income'
        else:
            category = np.random.choice(list(transaction_types.keys()))
        
        all_categories.append(category)
        all_descriptions.append(np.random.choice(transaction_types[category]))
    
    # Generate amounts based on category
    def generate_amount(category):
        if category == 'income':
            return np.random.uniform(1000, 5000)
        elif category == 'housing':
            return -np.random.uniform(800, 2000)
        elif category == 'food':
            return -np.random.uniform(10, 200)
        elif category == 'transport':
            return -np.random.uniform(20, 150)
        elif category == 'utilities':
            return -np.random.uniform(50, 300)
        elif category == 'entertainment':
            return -np.random.uniform(10, 100)
        elif category == 'shopping':
            return -np.random.uniform(20, 300)
        else:  # health
            return -np.random.uniform(20, 500)
    
    amounts = [generate_amount(cat) for cat in all_categories]
    
    # Create DataFrame
    df = pd.DataFrame({
        'transaction_date': dates,
        'description': all_descriptions,
        'amount': amounts,
        'category': all_categories
    })
    
    return df

def test_transaction_processing_performance(large_transaction_dataset, tmp_path):
    """Test performance of transaction processing"""
    print("\n===== Transaction Processing Performance Test =====")
    
    # Save the dataset to CSV for realistic file I/O testing
    csv_path = tmp_path / "large_transactions.csv"
    large_transaction_dataset.to_csv(csv_path, index=False)
    
    print(f"Dataset size: {len(large_transaction_dataset)} transactions")
    print(f"CSV file size: {os.path.getsize(csv_path) / (1024*1024):.2f} MB")
    
    # Initialize processor
    processor = TransactionProcessor()
    
    # Test loading performance
    start_time = time.time()
    df = processor.load_transactions(csv_path)
    load_time = time.time() - start_time
    print(f"CSV loading time: {load_time:.4f} seconds")
    
    # Profile cleaning performance
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.time()
    cleaned_df = processor.clean_transactions(df)
    clean_time = time.time() - start_time
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    print(f"\nProfiling results for clean_transactions:")
    print(s.getvalue())
    print(f"Transaction cleaning time: {clean_time:.4f} seconds")
    
    # Test categorization performance
    start_time = time.time()
    categorized_df = processor.simple_categorize(cleaned_df)
    categorize_time = time.time() - start_time
    print(f"Transaction categorization time: {categorize_time:.4f} seconds")
    
    # Test optimization performance
    start_time = time.time()
    optimized_df = optimize_dataframe(categorized_df, verbose=True)
    optimize_time = time.time() - start_time
    print(f"DataFrame optimization time: {optimize_time:.4f} seconds")
    
    # Test parallel processing performance
    def simple_process(chunk_df):
        # A simple processing function to test parallelization
        chunk_df['amount_abs'] = chunk_df['amount'].abs()
        chunk_df['month'] = pd.DatetimeIndex(chunk_df['transaction_date']).month
        chunk_df['day_of_week'] = pd.DatetimeIndex(chunk_df['transaction_date']).dayofweek
        return chunk_df
    
    start_time = time.time()
    processed_df = simple_process(optimized_df)
    serial_time = time.time() - start_time
    print(f"Serial processing time: {serial_time:.4f} seconds")
    
    # Test parallel processing with different numbers of cores
    results = []
    core_options = [2, 4, 8, 16] if os.cpu_count() >= 16 else [2, 4, os.cpu_count()]
    
    for cores in core_options:
        if cores > os.cpu_count():
            continue
            
        start_time = time.time()
        parallel_df = parallelize_dataframe(optimized_df, simple_process, n_cores=cores)
        parallel_time = time.time() - start_time
        
        speedup = serial_time / parallel_time
        efficiency = speedup / cores
        
        results.append({
            'cores': cores,
            'time': parallel_time,
            'speedup': speedup,
            'efficiency': efficiency
        })
        
        print(f"Parallel processing with {cores} cores: {parallel_time:.4f} seconds (speedup: {speedup:.2f}x)")
    
    # Create visualization directory
    viz_dir = tmp_path / "performance_viz"
    viz_dir.mkdir(exist_ok=True)
    
    # Plot scaling results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot([r['cores'] for r in results], [r['speedup'] for r in results], 'o-', linewidth=2)
    plt.axline((0, 0), (1, 1), color='gray', linestyle='--', label='Linear speedup')
    plt.xlabel('Number of cores')
    plt.ylabel('Speedup')
    plt.title('Parallel Speedup')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot([r['cores'] for r in results], [r['efficiency'] for r in results], 'o-', linewidth=2)
    plt.axhline(y=1.0, color='gray', linestyle='--', label='Perfect efficiency')
    plt.xlabel('Number of cores')
    plt.ylabel('Efficiency')
    plt.title('Parallel Efficiency')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(viz_dir / "parallel_scaling.png")
    
    # Plot processing time breakdown
    times = [load_time, clean_time, categorize_time, optimize_time]
    labels = ['Data Loading', 'Data Cleaning', 'Categorization', 'Optimization']
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, times)
    plt.ylabel('Time (seconds)')
    plt.title('Transaction Processing Performance Breakdown')
    plt.grid(axis='y')
    plt.savefig(viz_dir / "processing_breakdown.png")
    
    print(f"\nPerformance visualizations saved to: {viz_dir}")
    
    # Summary
    total_time = load_time + clean_time + categorize_time + optimize_time
    transactions_per_second = len(large_transaction_dataset) / total_time
    
    print("\n===== Performance Summary =====")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Transactions per second: {transactions_per_second:.1f}")
    print(f"Dataset size: {len(large_transaction_dataset)} rows")
    
    # Check maximum memory usage
    import psutil
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"Maximum memory usage: {memory_usage:.2f} MB")
    
    # Assertions to verify the test completed successfully
    assert len(categorized_df) == len(large_transaction_dataset)
    assert 'category' in categorized_df.columns
    assert not categorized_df['amount'].isna().any()