# app/models/forecasting/lstm_forecaster.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings

from app.models.forecasting.base_forecaster import BaseForecaster

class LSTMForecaster(BaseForecaster):
    """
    LSTM forecasting model implementation.
    Uses Long Short-Term Memory (LSTM) neural networks for time series forecasting.
    """
    
    def __init__(self, name: str = "lstm"):
        """
        Initialize the LSTM forecaster.
        
        Args:
            name: Name of the forecaster
        """
        super().__init__(name=name)
        self.model_params = {
            'sequence_length': 10,  # Number of past time steps to use as input
            'lstm_units': 50,       # Number of LSTM cells
            'dropout_rate': 0.2,    # Dropout rate for regularization
            'epochs': 50,           # Number of training epochs
            'batch_size': 32,       # Batch size for training
            'validation_split': 0.2, # Portion of data to use for validation
            'patience': 10          # Early stopping patience
        }
        self.scalers = {}           # Store scalers for each category
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and target values for LSTM training.
        
        Args:
            data: Time series data as numpy array
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) where X is the input sequences and y is the target values
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def _build_model(self, sequence_length: int) -> tf.keras.Model:
        """
        Build the LSTM model architecture.
        
        Args:
            sequence_length: Length of input sequences
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(
            units=self.model_params['lstm_units'],
            return_sequences=True,
            input_shape=(sequence_length, 1)
        ))
        model.add(Dropout(self.model_params['dropout_rate']))
        
        # Second LSTM layer
        model.add(LSTM(
            units=self.model_params['lstm_units'],
            return_sequences=False
        ))
        model.add(Dropout(self.model_params['dropout_rate']))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def fit(self, time_series_data: Dict[str, pd.DataFrame], 
           value_col: str = 'amount_sum', 
           date_col: str = 'transaction_date',
           categories: Optional[List[str]] = None,
           **kwargs) -> None:
        """
        Fit LSTM models on the time series data.
        
        Args:
            time_series_data: Dictionary mapping category names to time series DataFrames
            value_col: Column name for the target values
            date_col: Column name for dates
            categories: List of categories to fit models for (if None, fit for all categories)
            **kwargs: Additional parameters for LSTM model
        """
        self.models = {}
        self.forecasts = {}
        self.scalers = {}
        
        # Update model parameters if provided
        if kwargs:
            for key, value in kwargs.items():
                if key in self.model_params:
                    self.model_params[key] = value
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Process each time series
        for category, df in time_series_data.items():
            if categories is not None and category not in categories:
                continue
                
            # Prepare the time series
            ts_df = df.copy()
            
            # Ensure date column is datetime and set as index
            if date_col in ts_df.columns:
                ts_df[date_col] = pd.to_datetime(ts_df[date_col])
                ts_df.set_index(date_col, inplace=True)
            
            # Sort by date
            ts_df = ts_df.sort_index()
            
            # Extract the target series
            if value_col in ts_df.columns:
                series = ts_df[value_col].values.reshape(-1, 1)
            else:
                print(f"Warning: Column '{value_col}' not found in DataFrame for category '{category}'")
                continue
            
            # Check if we have enough data
            if len(series) < self.model_params['sequence_length'] * 3:
                print(f"Warning: Not enough data for category '{category}' ({len(series)} points)")
                continue
            
            try:
                # Normalize the data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(series)
                self.scalers[category] = scaler
                
                # Create sequences for LSTM
                sequence_length = self.model_params['sequence_length']
                X, y = self._create_sequences(scaled_data, sequence_length)
                
                # Build the model
                model = self._build_model(sequence_length)
                
                # Set up early stopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=self.model_params['patience'],
                    restore_best_weights=True
                )
                
                # Train the model
                model.fit(
                    X, y,
                    epochs=self.model_params['epochs'],
                    batch_size=self.model_params['batch_size'],
                    validation_split=self.model_params['validation_split'],
                    callbacks=[early_stopping],
                    verbose=0  # Set to 1 for progress bar
                )
                
                # Store the fitted model
                self.models[category] = {
                    'model': model,
                    'last_values': scaled_data[-sequence_length:].reshape(1, sequence_length, 1),
                    'last_date': ts_df.index[-1],
                    'sequence_length': sequence_length,
                    'freq': ts_df.index.freq or pd.infer_freq(ts_df.index) or 'D'
                }
                
                print(f"Successfully fitted LSTM model for category '{category}'")
                
            except Exception as e:
                print(f"Error fitting LSTM model for category '{category}': {str(e)}")
        
        self.is_fitted = len(self.models) > 0
        
        if self.is_fitted:
            print(f"Successfully fitted {len(self.models)} LSTM models")
        else:
            print("Failed to fit any LSTM models")
    
    def predict(self, horizon: int = 30, 
               categories: Optional[List[str]] = None,
               monte_carlo_simulations: int = 0,
               **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for the specified horizon.
        
        Args:
            horizon: Number of future time periods to forecast
            categories: List of categories to generate forecasts for (if None, forecast for all categories)
            monte_carlo_simulations: Number of simulations for uncertainty estimation
            **kwargs: Additional parameters for forecast
            
        Returns:
            Dictionary mapping category names to DataFrames with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self.forecasts = {}
        
        if categories is None:
            categories = list(self.models.keys())
        
        for category in categories:
            if category not in self.models:
                print(f"Warning: No model found for category '{category}'")
                continue
                
            if category not in self.scalers:
                print(f"Warning: No scaler found for category '{category}'")
                continue
                
            model_info = self.models[category]
            model = model_info['model']
            last_values = model_info['last_values'].copy()
            last_date = model_info['last_date']
            freq = model_info['freq']
            scaler = self.scalers[category]
            
            try:
                # Generate date range for forecast
                forecast_index = pd.date_range(
                    start=last_date,
                    periods=horizon + 1,
                    freq=freq
                )[1:]  # Exclude the last known date
                
                # Generate forecast
                predictions = []
                
                if monte_carlo_simulations > 0:
                    # Monte Carlo simulation for uncertainty estimation
                    all_simulations = []
                    
                    for _ in range(monte_carlo_simulations):
                        current_values = last_values.copy()
                        simulation = []
                        
                        for _ in range(horizon):
                            # Predict next value
                            pred = model.predict(current_values, verbose=0)[0][0]
                            simulation.append(pred)
                            
                            # Update input sequence
                            current_values = np.roll(current_values, -1, axis=1)
                            current_values[0, -1, 0] = pred
                        
                        all_simulations.append(simulation)
                    
                    # Calculate mean and confidence intervals
                    all_simulations = np.array(all_simulations)
                    mean_pred = np.mean(all_simulations, axis=0)
                    lower_bound = np.percentile(all_simulations, 2.5, axis=0)
                    upper_bound = np.percentile(all_simulations, 97.5, axis=0)
                    
                    # Inverse transform the predictions
                    predictions = scaler.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
                    lower_bounds = scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
                    upper_bounds = scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()
                    
                    # Create forecast DataFrame
                    forecast_df = pd.DataFrame({
                        'ds': forecast_index,
                        'amount': predictions,
                        'lower_bound': lower_bounds,
                        'upper_bound': upper_bounds
                    })
                else:
                    # Simple prediction without uncertainty estimation
                    current_values = last_values.copy()
                    
                    for _ in range(horizon):
                        # Predict next value
                        pred = model.predict(current_values, verbose=0)[0][0]
                        predictions.append(pred)
                        
                        # Update input sequence
                        current_values = np.roll(current_values, -1, axis=1)
                        current_values[0, -1, 0] = pred
                    
                    # Inverse transform the predictions
                    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                    
                    # Create forecast DataFrame
                    forecast_df = pd.DataFrame({
                        'ds': forecast_index,
                        'amount': predictions
                    })
                
                # Store the forecast
                self.forecasts[category] = forecast_df
                
                print(f"Generated {horizon} step forecast for category '{category}'")
                
            except Exception as e:
                print(f"Error generating forecast for category '{category}': {str(e)}")
        
        return self.forecasts
    
    def save_model(self, output_dir: str = 'app/models/forecasting/saved_models') -> None:
        """
        Save trained LSTM models to disk.
        
        Args:
            output_dir: Directory to save the models
        """
        if not self.is_fitted:
            raise ValueError("No fitted models to save")
            
        os.makedirs(output_dir, exist_ok=True)
        
        for category, model_info in self.models.items():
            # Create safe filename
            safe_category = category.replace(' ', '_').lower()
            model_path = os.path.join(output_dir, f"{self.name}_{safe_category}_model")
            
            try:
                # Save the Keras model
                model_info['model'].save(model_path)
                
                # Save other model info and scaler
                info_dict = {
                    'last_values': model_info['last_values'],
                    'last_date': model_info['last_date'],
                    'sequence_length': model_info['sequence_length'],
                    'freq': model_info['freq'],
                    'scaler': self.scalers.get(category)
                }
                
                joblib.dump(info_dict, f"{model_path}_info.pkl")
                
                print(f"Saved model for category '{category}' to {model_path}")
            except Exception as e:
                print(f"Error saving model for category '{category}': {str(e)}")
    
    def load_model(self, input_dir: str = 'app/models/forecasting/saved_models') -> None:
        """
        Load trained LSTM models from disk.
        
        Args:
            input_dir: Directory to load the models from
        """
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory '{input_dir}' does not exist")
            
        model_files = [f for f in os.listdir(input_dir) if f.startswith(f"{self.name}_") and f.endswith("_model")]
        
        if not model_files:
            raise ValueError(f"No {self.name} model files found in {input_dir}")
            
        self.models = {}
        self.scalers = {}
        
        for model_file in model_files:
            model_path = os.path.join(input_dir, model_file)
            info_path = f"{model_path}_info.pkl"
            
            try:
                # Extract category from filename
                category = model_file.replace(f"{self.name}_", "").replace("_model", "").replace("_", " ")
                
                # Load the Keras model
                keras_model = load_model(model_path)
                
                # Load other model info and scaler
                info_dict = joblib.load(info_path)
                
                # Store the model
                self.models[category] = {
                    'model': keras_model,
                    'last_values': info_dict['last_values'],
                    'last_date': info_dict['last_date'],
                    'sequence_length': info_dict['sequence_length'],
                    'freq': info_dict['freq']
                }
                
                # Store the scaler
                self.scalers[category] = info_dict['scaler']
                
                print(f"Loaded model for category '{category}' from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {str(e)}")
        
        self.is_fitted = len(self.models) > 0
        
        if self.is_fitted:
            print(f"Successfully loaded {len(self.models)} LSTM models")
        else:
            print("Failed to load any LSTM models")