"""
Performance optimization utilities for the Financial Health Assistant
"""
import time
import functools
import logging
import cProfile
import pstats
import io
import os
import pandas as pd
import numpy as np
from typing import Callable, Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timeit(func):
    """
    Decorator to measure and log the execution time of functions
    
    Args:
        func: The function to measure
        
    Returns:
        Wrapped function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {elapsed:.4f} seconds")
        return result
    return wrapper

def profile_function(func):
    """
    Decorator to profile a function using cProfile
    
    Args:
        func: The function to profile
        
    Returns:
        Wrapped function that profiles execution
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Print top 20 functions by time
        logger.info(f"Profile for '{func.__name__}':\n{s.getvalue()}")
        
        return result
    return wrapper

def memory_usage(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate the memory usage of a pandas DataFrame
    
    Args:
        dataframe: The DataFrame to analyze
        
    Returns:
        Dictionary with memory usage information
    """
    memory_usage_info = {}
    memory_usage_info['total_memory_mb'] = dataframe.memory_usage(deep=True).sum() / 1024 / 1024
    memory_usage_info['rows'] = len(dataframe)
    memory_usage_info['columns'] = len(dataframe.columns)
    memory_usage_info['memory_per_row_kb'] = (memory_usage_info['total_memory_mb'] * 1024) / len(dataframe)
    
    # Memory usage by column
    column_memory = dataframe.memory_usage(deep=True)
    column_memory_dict = {}
    for column, memory in zip(column_memory.index, column_memory):
        if column != 'Index':
            column_memory_dict[column] = memory / 1024 / 1024  # MB
    
    memory_usage_info['column_memory_mb'] = column_memory_dict
    
    return memory_usage_info

def optimize_dataframe(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Optimize the memory usage of a pandas DataFrame
    
    Args:
        df: The DataFrame to optimize
        verbose: Whether to print optimization details
        
    Returns:
        Optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    if verbose:
        logger.info(f"Memory usage before optimization: {start_mem:.2f} MB")
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        # Determine the min and max values
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Convert to the smallest possible integer type
        if col_min >= 0:
            if col_max < 2**8:
                df[col] = df[col].astype(np.uint8)
            elif col_max < 2**16:
                df[col] = df[col].astype(np.uint16)
            elif col_max < 2**32:
                df[col] = df[col].astype(np.uint32)
            else:
                df[col] = df[col].astype(np.uint64)
        else:
            if col_min > -2**7 and col_max < 2**7:
                df[col] = df[col].astype(np.int8)
            elif col_min > -2**15 and col_max < 2**15:
                df[col] = df[col].astype(np.int16)
            elif col_min > -2**31 and col_max < 2**31:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
    
    # Optimize float columns
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimize string columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    reduction = (start_mem - end_mem) / start_mem * 100
    
    if verbose:
        logger.info(f"Memory usage after optimization: {end_mem:.2f} MB")
        logger.info(f"Memory reduced by {reduction:.2f}%")
    
    return df

def parallelize_dataframe(df: pd.DataFrame, func: Callable, n_cores: Optional[int] = None) -> pd.DataFrame:
    """
    Apply a function to a DataFrame in parallel
    
    Args:
        df: The DataFrame to process
        func: The function to apply
        n_cores: Number of cores to use (default: None, uses all available)
        
    Returns:
        Processed DataFrame
    """
    import multiprocessing as mp
    
    if n_cores is None:
        n_cores = mp.cpu_count()
    
    # Split dataframe into n_cores chunks
    df_split = np.array_split(df, n_cores)
    
    # Process each chunk in parallel
    with mp.Pool(n_cores) as pool:
        df = pd.concat(pool.map(func, df_split))
    
    return df

def cache_to_disk(filename: str, directory: str = 'cache'):
    """
    Decorator to cache function results to disk
    
    Args:
        filename: Base filename for the cache
        directory: Directory to store cache files
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Create a unique cache filename based on function args
            args_str = '_'.join([str(arg) for arg in args])
            kwargs_str = '_'.join([f"{k}_{v}" for k, v in kwargs.items()])
            cache_file = os.path.join(directory, f"{filename}_{args_str}_{kwargs_str}.pkl")
            
            # If cache file exists, load and return it
            if os.path.exists(cache_file):
                try:
                    result = pd.read_pickle(cache_file)
                    logger.info(f"Loaded cached result from {cache_file}")
                    return result
                except Exception as e:
                    logger.warning(f"Failed to load cache: {str(e)}")
            
            # Otherwise, compute the result and cache it
            result = func(*args, **kwargs)
            
            try:
                pd.to_pickle(result, cache_file)
                logger.info(f"Cached result to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache result: {str(e)}")
            
            return result
        return wrapper
    return decorator