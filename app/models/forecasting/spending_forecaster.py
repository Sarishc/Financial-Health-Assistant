import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error
import os

class SpendingForecaster:
    """Time-series forecasting model for spending prediction"""
    
    def __init__(self):
        """Initialize the forecasting model"""
        self.models = {}  # Category-specific models
        self.forecast_days = 30  # Default forecast horizon
    
    def prepare_time_series(self, df: pd.DataFrame,
                           date_col: str = 'transaction_date',
                           amount_col: str = 'amount',
                           category_col: str = 'category') -> Dict[str, pd.DataFrame]:
        """
        Transform transaction data into time series format for forecasting
        
        Args:
            df: DataFrame containing transaction data
            date_col: Column name for transaction dates
            amount_col: Column name for transaction amounts
            category_col: Column name for transaction categories
            
        Returns:
            Dictionary of category-specific time series DataFrames
        """
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Drop rows with missing dates
        df = df.dropna(subset=[date_col])
        
        # For forecasting, we need to aggregate by day
        # Expenses are typically negative, let's make them positive for forecasting
        if amount_col in df.columns:
            # Handle different transaction data formats
            if df[amount_col].min() < 0:
                # If negative values exist, assume they are expenses
                df_expenses = df[df[amount_col] < 0].copy()
                df_expenses[amount_col] = df_expenses[amount_col].abs()
            else:
                # If only positive values, try to identify expenses vs. income
                if 'withdrawal' in df.columns:
                    # Use withdrawal column for expenses
                    df_expenses = df[df['withdrawal'] > 0].copy()
                    df_expenses[amount_col] = df_expenses['withdrawal']
                else:
                    # Assume all are expenses
                    df_expenses = df.copy()
        elif 'withdrawal' in df.columns:
            # If amount column doesn't exist but withdrawal does
            df_expenses = df[df['withdrawal'] > 0].copy()
            df_expenses[amount_col] = df_expenses['withdrawal']
        else:
            raise ValueError("Cannot identify expense amounts in the data")
        
        # Group by date and category
        if category_col in df.columns:
            # Group by date and category
            daily_spending = df_expenses.groupby([
                pd.Grouper(key=date_col, freq='D'),
                category_col
            ])[amount_col].sum().reset_index()
            
            # Create dictionary of category-specific time series
            category_dfs = {}
            
            for category in daily_spending[category_col].unique():
                category_data = daily_spending[daily_spending[category_col] == category]
                category_data = category_data.set_index(date_col)
                # Ensure the time series has no gaps
                date_range = pd.date_range(
                    start=category_data.index.min(),
                    end=category_data.index.max(),
                    freq='D'
                )
                category_ts = pd.DataFrame(index=date_range)
                category_ts[amount_col] = category_data[amount_col]
                category_ts = category_ts.fillna(0)
                
                category_dfs[category] = category_ts
        else:
            # If no category column, create a single time series for all expenses
            daily_spending = df_expenses.groupby(
                pd.Grouper(key=date_col, freq='D')
            )[amount_col].sum().reset_index()
            
            daily_spending = daily_spending.set_index(date_col)
            
            # Ensure the time series has no gaps
            date_range = pd.date_range(
                start=daily_spending.index.min(),
                end=daily_spending.index.max(),
                freq='D'
            )
            daily_ts = pd.DataFrame(index=date_range)
            daily_ts[amount_col] = daily_spending[amount_col]
            daily_ts = daily_ts.fillna(0)
            
            category_dfs = {'all': daily_ts}
        
        return category_dfs
    
    def train_arima_model(self, time_series: pd.DataFrame, 
                         column: str = 'amount') -> Tuple[Any, dict]:
        """
        Train an ARIMA model for a specific time series
        
        Args:
            time_series: DataFrame containing the time series data
            column: Column name of the values to forecast
            
        Returns:
            Tuple of (trained model, parameters)
        """
        # Check if we have enough data
        if len(time_series) < 30:
            print(f"Warning: Time series has only {len(time_series)} data points, which may be insufficient")
        
        # Try to find best ARIMA parameters (simplified approach)
        # For production, use auto_arima from pmdarima
        try:
            # Simple parameter search
            best_model = None
            best_aic = float('inf')
            best_params = (1, 0, 1)  # Default parameters
            
            # Try a few parameter combinations
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(time_series[column].astype(float), order=(p, d, q))
                            model_fit = model.fit()
                            aic = model_fit.aic
                            
                            if aic < best_aic:
                                best_aic = aic
                                best_model = model_fit
                                best_params = (p, d, q)
                        except:
                            continue
            
            if best_model is None:
                # If all parameter combinations fail, try default
                model = ARIMA(time_series[column].astype(float), order=(1, 0, 1))
                best_model = model.fit()
                best_params = (1, 0, 1)
            
            return best_model, {'order': best_params}
        except Exception as e:
            print(f"Error training ARIMA model: {str(e)}")
            # Fallback to very simple model
            model = ARIMA(time_series[column].astype(float), order=(1, 0, 0))
            model_fit = model.fit()
            return model_fit, {'order': (1, 0, 0)}
    
    def train_models(self, category_dfs: Dict[str, pd.DataFrame],
                    column: str = 'amount') -> None:
        """
        Train forecasting models for each spending category
        
        Args:
            category_dfs: Dictionary of category-specific time series
            column: Column name of the values to forecast
        """
        self.models = {}
        
        for category, df in category_dfs.items():
            print(f"Training model for category: {category}")
            
            # Skip if not enough data
            if len(df) < 14:  # Need at least 2 weeks of data
                print(f"  Skipping category '{category}' due to insufficient data ({len(df)} points)")
                continue
            
            # Train ARIMA model
            try:
                model, params = self.train_arima_model(df, column)
                self.models[category] = {
                    'model': model,
                    'params': params,
                    'last_date': df.index.max()
                }
                print(f"  Trained model for '{category}' with parameters {params['order']}")
            except Exception as e:
                print(f"  Error training model for '{category}': {str(e)}")
    
    def forecast(self, days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Generate spending forecasts for each category
        
        Args:
            days: Number of days to forecast
            
        Returns:
            Dictionary of category-specific forecasts
        """
        if not self.models:
            raise RuntimeError("No trained models available. Call train_models first.")
        
        forecasts = {}
        self.forecast_days = days
        
        for category, model_info in self.models.items():
            model = model_info['model']
            last_date = model_info['last_date']
            
            # Generate forecast
            try:
                forecast = model.forecast(steps=days)
                
                # Create DataFrame with dates
                date_range = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=days,
                    freq='D'
                )
                
                forecast_df = pd.DataFrame({
                    'date': date_range,
                    'amount': forecast,
                    'category': category
                })
                
                # Calculate confidence intervals (using model predictions)
                if hasattr(model, 'get_prediction'):
                    pred = model.get_prediction(start=len(model.endog), end=len(model.endog) + days - 1)
                    pred_ci = pred.conf_int(alpha=0.05)  # 95% confidence interval
                    forecast_df['lower_bound'] = pred_ci.iloc[:, 0]
                    forecast_df['upper_bound'] = pred_ci.iloc[:, 1]
                else:
                    # Simple approximation if confidence intervals not available
                    forecast_df['lower_bound'] = forecast * 0.8
                    forecast_df['upper_bound'] = forecast * 1.2
                
                forecasts[category] = forecast_df
                print(f"Generated {days}-day forecast for '{category}'")
            except Exception as e:
                print(f"Error generating forecast for '{category}': {str(e)}")
        
        return forecasts
    
    def visualize_forecast(self, category: str, forecast_df: pd.DataFrame, 
                          historical_df: Optional[pd.DataFrame] = None,
                          column: str = 'amount',
                          output_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the forecast for a specific category
        
        Args:
            category: Category name
            forecast_df: DataFrame containing the forecast
            historical_df: DataFrame containing historical data (optional)
            column: Column name of the values
            output_path: Path to save the visualization (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data if available
        if historical_df is not None:
            ax.plot(historical_df.index, historical_df[column], 'k.-', label='Historical')
        
        # Plot forecast
        ax.plot(forecast_df['date'], forecast_df['amount'], 'b.-', label='Forecast')
        
        # Plot confidence intervals if available
        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
            ax.fill_between(
                forecast_df['date'],
                forecast_df['lower_bound'],
                forecast_df['upper_bound'],
                color='b', alpha=0.2, label='95% Confidence Interval'
            )
        
        # Set title and labels
        ax.set_title(f'Spending Forecast for {category.capitalize()}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Amount')
        
        # Add legend
        ax.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
        
        return fig
    
    def save_models(self, base_path: str) -> None:
        """
        Save all trained models to disk
        
        Args:
            base_path: Base directory to save models
        """
        os.makedirs(base_path, exist_ok=True)
        
        for category, model_info in self.models.items():
            # Create category-specific filename
            filename = f"{category.replace(' ', '_')}_forecast_model.pkl"
            filepath = os.path.join(base_path, filename)
            
            try:
                # Save the model using joblib
                joblib.dump(model_info, filepath)
                print(f"Saved model for '{category}' to {filepath}")
            except Exception as e:
                print(f"Error saving model for '{category}': {str(e)}")
    
    def load_models(self, base_path: str) -> None:
        """
        Load all saved models from disk
        
        Args:
            base_path: Base directory containing saved models
        """
        if not os.path.exists(base_path):
            raise ValueError(f"Model directory not found: {base_path}")
        
        self.models = {}
        
        # Find all model files
        model_files = [f for f in os.listdir(base_path) if f.endswith('_forecast_model.pkl')]
        
        for filename in model_files:
            filepath = os.path.join(base_path, filename)
            
            try:
                # Extract category from filename
                category = filename.replace('_forecast_model.pkl', '').replace('_', ' ')
                
                # Load the model
                model_info = joblib.load(filepath)
                self.models[category] = model_info
                print(f"Loaded model for '{category}' from {filepath}")
            except Exception as e:
                print(f"Error loading model from {filepath}: {str(e)}")