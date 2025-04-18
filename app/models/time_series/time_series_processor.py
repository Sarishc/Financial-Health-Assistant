# app/models/time_series/time_series_processor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

class TimeSeriesProcessor:
    """
    Processes transaction data into time series format for analysis and forecasting.
    Handles temporal feature extraction, aggregation, and visualization.
    """
    
    def __init__(self):
        """Initialize the time series processor"""
        self.time_periods = ['D', 'W', 'M', 'Q', 'Y']  # Day, Week, Month, Quarter, Year
        self.visualization_dir = 'notebooks/visualizations/time_series'
        os.makedirs(self.visualization_dir, exist_ok=True)
    
    def convert_to_time_series(self, df: pd.DataFrame, 
                              date_col: str = 'transaction_date',
                              amount_col: str = 'amount',
                              category_col: Optional[str] = 'category',
                              freq: str = 'D') -> pd.DataFrame:
        """
        Convert transaction data to time series format
        
        Args:
            df: DataFrame containing transaction data
            date_col: Column name for transaction dates
            amount_col: Column name for transaction amounts
            category_col: Column name for transaction categories (optional)
            freq: Frequency for time series ('D'=daily, 'W'=weekly, 'M'=monthly)
            
        Returns:
            DataFrame in time series format
        """
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Remove rows with missing dates
        df = df.dropna(subset=[date_col])
        
        # Group by date and optionally by category
        if category_col and category_col in df.columns:
            # Group by both date and category
            grouped = df.groupby([pd.Grouper(key=date_col, freq=freq), category_col])
            time_series = grouped.agg({
                amount_col: ['sum', 'count', 'mean']
            })
            
            # Flatten column names
            time_series.columns = ['_'.join(col).strip() for col in time_series.columns.values]
            
            # Reset index for easier handling
            time_series = time_series.reset_index()
        else:
            # Group only by date
            grouped = df.groupby(pd.Grouper(key=date_col, freq=freq))
            time_series = grouped.agg({
                amount_col: ['sum', 'count', 'mean']
            })
            
            # Flatten column names
            time_series.columns = ['_'.join(col).strip() for col in time_series.columns.values]
            
            # Reset index for easier handling
            time_series = time_series.reset_index()
        
        return time_series
    
    def extract_temporal_features(self, df: pd.DataFrame, 
                                date_col: str = 'transaction_date') -> pd.DataFrame:
        """
        Extract temporal features from datetime column
        
        Args:
            df: DataFrame containing transaction data with datetime column
            date_col: Column name for transaction dates
            
        Returns:
            DataFrame with added temporal features
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Ensure date column is datetime
        result_df[date_col] = pd.to_datetime(result_df[date_col], errors='coerce')
        
        # Extract basic time components
        result_df['year'] = result_df[date_col].dt.year
        result_df['quarter'] = result_df[date_col].dt.quarter
        result_df['month'] = result_df[date_col].dt.month
        result_df['day'] = result_df[date_col].dt.day
        result_df['dayofweek'] = result_df[date_col].dt.dayofweek  # Monday=0, Sunday=6
        result_df['is_weekend'] = result_df['dayofweek'].isin([5, 6]).astype(int)
        result_df['is_month_start'] = result_df[date_col].dt.is_month_start.astype(int)
        result_df['is_month_end'] = result_df[date_col].dt.is_month_end.astype(int)
        
        # Add seasonality features
        result_df['day_sin'] = np.sin(2 * np.pi * result_df['day'] / 31)
        result_df['day_cos'] = np.cos(2 * np.pi * result_df['day'] / 31)
        result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
        result_df['dayofweek_sin'] = np.sin(2 * np.pi * result_df['dayofweek'] / 7)
        result_df['dayofweek_cos'] = np.cos(2 * np.pi * result_df['dayofweek'] / 7)
        
        return result_df
    
    def create_lagged_features(self, df: pd.DataFrame, 
                              value_col: str,
                              lag_periods: List[int] = [1, 7, 30],
                              group_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create lagged features for time series analysis
        
        Args:
            df: DataFrame in time series format
            value_col: Column name for the value to lag
            lag_periods: List of periods to lag by
            group_col: Column name to group by (e.g., 'category')
            
        Returns:
            DataFrame with added lag features
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Ensure data is sorted by date
        if 'transaction_date' in result_df.columns:
            result_df = result_df.sort_values('transaction_date')
        
        # Create lagged features
        for lag in lag_periods:
            lag_col = f'{value_col}_lag_{lag}'
            
            if group_col and group_col in result_df.columns:
                # Create lags within each group
                result_df[lag_col] = result_df.groupby(group_col)[value_col].shift(lag)
            else:
                # Create lags across all data
                result_df[lag_col] = result_df[value_col].shift(lag)
        
        # Calculate pct change as well (if not too many NaNs)
        if len(result_df) > max(lag_periods) + 10:  # Only if we have enough data
            for lag in lag_periods:
                pct_change_col = f'{value_col}_pct_change_{lag}'
                
                if group_col and group_col in result_df.columns:
                    # Calculate pct change within each group
                    result_df[pct_change_col] = result_df.groupby(group_col)[value_col].pct_change(periods=lag)
                else:
                    # Calculate pct change across all data
                    result_df[pct_change_col] = result_df[value_col].pct_change(periods=lag)
        
        return result_df
    
    def create_rolling_features(self, df: pd.DataFrame,
                               value_col: str,
                               window_sizes: List[int] = [7, 14, 30],
                               group_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create rolling window features for time series analysis
        
        Args:
            df: DataFrame in time series format
            value_col: Column name for the value to create rolling features from
            window_sizes: List of window sizes for rolling calculations
            group_col: Column name to group by (e.g., 'category')
            
        Returns:
            DataFrame with added rolling features
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Ensure data is sorted by date
        if 'transaction_date' in result_df.columns:
            result_df = result_df.sort_values('transaction_date')
        
        # Create rolling features
        for window in window_sizes:
            # Rolling mean
            mean_col = f'{value_col}_rolling_mean_{window}'
            
            if group_col and group_col in result_df.columns:
                # Calculate rolling mean within each group
                result_df[mean_col] = result_df.groupby(group_col)[value_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
            else:
                # Calculate rolling mean across all data
                result_df[mean_col] = result_df[value_col].rolling(window=window, min_periods=1).mean()
            
            # Rolling standard deviation
            std_col = f'{value_col}_rolling_std_{window}'
            
            if group_col and group_col in result_df.columns:
                # Calculate rolling std within each group
                result_df[std_col] = result_df.groupby(group_col)[value_col].transform(
                    lambda x: x.rolling(window=window, min_periods=3).std()
                )
            else:
                # Calculate rolling std across all data
                result_df[std_col] = result_df[value_col].rolling(window=window, min_periods=3).std()
        
        return result_df
    
    def create_seasonal_features(self, df: pd.DataFrame,
                               date_col: str = 'transaction_date') -> pd.DataFrame:
        """
        Create seasonal features for time series analysis
        
        Args:
            df: DataFrame containing transaction data
            date_col: Column name for transaction dates
            
        Returns:
            DataFrame with added seasonal features
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Ensure date column is datetime
        result_df[date_col] = pd.to_datetime(result_df[date_col], errors='coerce')
        
        # Define seasons (meteorological seasons in Northern Hemisphere)
        def get_season(month):
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'fall'
        
        # Add season column
        result_df['season'] = result_df[date_col].dt.month.apply(get_season)
        
        # Add holiday indicators (simplified approach)
        result_df['is_holiday'] = 0
        
        # Mark common US holidays (simplified)
        for date in result_df[date_col]:
            month, day = date.month, date.day
            
            # New Year's Day
            if month == 1 and day == 1:
                result_df.loc[result_df[date_col] == date, 'is_holiday'] = 1
            
            # Independence Day
            elif month == 7 and day == 4:
                result_df.loc[result_df[date_col] == date, 'is_holiday'] = 1
            
            # Christmas
            elif month == 12 and day == 25:
                result_df.loc[result_df[date_col] == date, 'is_holiday'] = 1
        
        return result_df
    
    def detect_outliers(self, df: pd.DataFrame,
                       value_col: str,
                       method: str = 'zscore',
                       threshold: float = 3.0,
                       group_col: Optional[str] = None) -> pd.DataFrame:
        """
        Detect outliers in time series data
        
        Args:
            df: DataFrame in time series format
            value_col: Column name for the value to check for outliers
            method: Method for outlier detection ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            group_col: Column name to group by (e.g., 'category')
            
        Returns:
            DataFrame with added outlier indicator
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Create outlier indicator column
        outlier_col = f'{value_col}_is_outlier'
        result_df[outlier_col] = 0
        
        if method == 'zscore':
            if group_col and group_col in result_df.columns:
                # Calculate z-scores within each group
                for group, group_df in result_df.groupby(group_col):
                    mean = group_df[value_col].mean()
                    std = group_df[value_col].std()
                    
                    if std > 0:  # Avoid division by zero
                        z_scores = (group_df[value_col] - mean) / std
                        outlier_mask = abs(z_scores) > threshold
                        
                        # Update outlier indicator
                        result_df.loc[outlier_mask.index[outlier_mask], outlier_col] = 1
            else:
                # Calculate z-scores across all data
                mean = result_df[value_col].mean()
                std = result_df[value_col].std()
                
                if std > 0:  # Avoid division by zero
                    z_scores = (result_df[value_col] - mean) / std
                    result_df[outlier_col] = (abs(z_scores) > threshold).astype(int)
        
        elif method == 'iqr':
            if group_col and group_col in result_df.columns:
                # Calculate IQR within each group
                for group, group_df in result_df.groupby(group_col):
                    q1 = group_df[value_col].quantile(0.25)
                    q3 = group_df[value_col].quantile(0.75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    
                    outlier_mask = (group_df[value_col] < lower_bound) | (group_df[value_col] > upper_bound)
                    
                    # Update outlier indicator
                    result_df.loc[outlier_mask.index[outlier_mask], outlier_col] = 1
            else:
                # Calculate IQR across all data
                q1 = result_df[value_col].quantile(0.25)
                q3 = result_df[value_col].quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                result_df[outlier_col] = ((result_df[value_col] < lower_bound) | 
                                         (result_df[value_col] > upper_bound)).astype(int)
        
        return result_df
    
    def visualize_time_series(self, df: pd.DataFrame, 
                             date_col: str = 'transaction_date',
                             value_col: str = 'amount_sum',
                             category_col: Optional[str] = None,
                             title: str = 'Time Series Visualization',
                             output_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize time series data
        
        Args:
            df: DataFrame in time series format
            date_col: Column name for dates
            value_col: Column name for values to plot
            category_col: Column name for categories (for grouped visualization)
            title: Plot title
            output_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(12, 6))
        
        if category_col and category_col in df.columns:
            # Plot for each category
            categories = df[category_col].unique()
            
            for category in categories:
                category_data = df[df[category_col] == category]
                plt.plot(category_data[date_col], category_data[value_col], label=category)
            
            plt.legend(title=category_col)
        else:
            # Plot all data
            plt.plot(df[date_col], df[value_col])
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(value_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        
        return plt.gcf()
    
    def visualize_spending_patterns(self, df: pd.DataFrame,
                                  date_col: str = 'transaction_date',
                                  amount_col: str = 'amount',
                                  category_col: Optional[str] = 'category',
                                  freq: str = 'M',
                                  output_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Generate comprehensive visualizations for spending patterns
        
        Args:
            df: DataFrame containing transaction data
            date_col: Column name for transaction dates
            amount_col: Column name for transaction amounts
            category_col: Column name for transaction categories
            freq: Frequency for aggregation ('D'=daily, 'W'=weekly, 'M'=monthly)
            output_dir: Directory to save plots (optional)
            
        Returns:
            Dictionary of {plot_name: matplotlib_figure}
        """
        if output_dir is None:
            output_dir = self.visualization_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create time series data
        time_series_df = self.convert_to_time_series(
            df, 
            date_col=date_col,
            amount_col=amount_col,
            category_col=category_col,
            freq=freq
        )
        
        # Store figures
        figures = {}
        
        # 1. Overall spending trend
        fig1 = plt.figure(figsize=(12, 6))
        
        if 'category' in time_series_df.columns:
            pivot_df = time_series_df.pivot(index='transaction_date', columns='category', values='amount_sum')
            pivot_df.plot(kind='line', ax=plt.gca())
        else:
            plt.plot(time_series_df['transaction_date'], time_series_df['amount_sum'])
        
        plt.title(f'Spending Trend ({freq} Frequency)')
        plt.xlabel('Date')
        plt.ylabel('Amount')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'spending_trend_{freq}.png')
        plt.savefig(output_path)
        figures['spending_trend'] = fig1
        
        # 2. Category breakdown
        if category_col and category_col in time_series_df.columns:
            fig2 = plt.figure(figsize=(12, 6))
            
            category_totals = time_series_df.groupby(category_col)['amount_sum'].sum().sort_values(ascending=False)
            category_totals.plot(kind='bar', ax=plt.gca())
            
            plt.title('Total Spending by Category')
            plt.xlabel('Category')
            plt.ylabel('Total Amount')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, 'category_breakdown.png')
            plt.savefig(output_path)
            figures['category_breakdown'] = fig2
        
        # 3. Monthly pattern
        fig3 = plt.figure(figsize=(12, 6))
        
        # Extract month from date
        time_series_df['month'] = pd.to_datetime(time_series_df['transaction_date']).dt.month
        
        if category_col and category_col in time_series_df.columns:
            monthly_pattern = time_series_df.groupby(['month', category_col])['amount_sum'].mean().reset_index()
            pivot_df = monthly_pattern.pivot(index='month', columns=category_col, values='amount_sum')
            pivot_df.plot(kind='bar', stacked=True, ax=plt.gca())
        else:
            monthly_pattern = time_series_df.groupby('month')['amount_sum'].mean()
            monthly_pattern.plot(kind='bar', ax=plt.gca())
        
        plt.title('Monthly Spending Pattern')
        plt.xlabel('Month')
        plt.ylabel('Average Amount')
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'monthly_pattern.png')
        plt.savefig(output_path)
        figures['monthly_pattern'] = fig3
        
        # 4. Day of week pattern
        fig4 = plt.figure(figsize=(12, 6))
        
        # Extract day of week from date
        time_series_df['dayofweek'] = pd.to_datetime(time_series_df['transaction_date']).dt.dayofweek
        
        if category_col and category_col in time_series_df.columns:
            dayofweek_pattern = time_series_df.groupby(['dayofweek', category_col])['amount_sum'].mean().reset_index()
            pivot_df = dayofweek_pattern.pivot(index='dayofweek', columns=category_col, values='amount_sum')
            pivot_df.plot(kind='bar', stacked=True, ax=plt.gca())
        else:
            dayofweek_pattern = time_series_df.groupby('dayofweek')['amount_sum'].mean()
            dayofweek_pattern.plot(kind='bar', ax=plt.gca())
        
        plt.title('Day of Week Spending Pattern')
        plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
        plt.ylabel('Average Amount')
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'dayofweek_pattern.png')
        plt.savefig(output_path)
        figures['dayofweek_pattern'] = fig4
        
        return figures
    
    def process_time_series_data(self, df: pd.DataFrame,
                              date_col: str = 'transaction_date',
                              amount_col: str = 'amount',
                              category_col: Optional[str] = 'category',
                              freq: str = 'D') -> pd.DataFrame:
        """
        Complete pipeline for time series processing
        
        Args:
            df: DataFrame containing transaction data
            date_col: Column name for transaction dates
            amount_col: Column name for transaction amounts
            category_col: Column name for transaction categories
            freq: Frequency for aggregation ('D'=daily, 'W'=weekly, 'M'=monthly)
            
        Returns:
            Processed DataFrame with time series features
        """
        # 1. Convert to time series format
        time_series_df = self.convert_to_time_series(
            df, 
            date_col=date_col,
            amount_col=amount_col,
            category_col=category_col,
            freq=freq
        )
        
        # 2. Extract temporal features
        time_series_df = self.extract_temporal_features(time_series_df, date_col='transaction_date')
        
        # 3. Create lagged features
        time_series_df = self.create_lagged_features(
            time_series_df,
            value_col='amount_sum',
            lag_periods=[1, 7, 30],
            group_col=category_col
        )
        
        # 4. Create rolling features
        time_series_df = self.create_rolling_features(
            time_series_df,
            value_col='amount_sum',
            window_sizes=[7, 14, 30],
            group_col=category_col
        )
        
        # 5. Create seasonal features
        time_series_df = self.create_seasonal_features(time_series_df, date_col='transaction_date')
        
        # 6. Detect outliers
        time_series_df = self.detect_outliers(
            time_series_df,
            value_col='amount_sum',
            method='zscore',
            threshold=3.0,
            group_col=category_col
        )
        
        return time_series_df