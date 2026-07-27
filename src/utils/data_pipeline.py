"""Data preprocessing pipeline for model training"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler
from .logger import logger
from .exceptions import ValidationError
from .config import Config

class DataPipeline:
    """Handle data preprocessing for machine learning models"""
    
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.feature_scalers: Dict[str, MinMaxScaler] = {}
        
    def prepare_training_data(self,
                            df: pd.DataFrame,
                            target_column: str = 'Close',
                            sequence_length: int = Config.MODEL_SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training
        
        Args:
            df: Input DataFrame
            target_column: Column to predict
            sequence_length: Length of input sequences
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) arrays for training
        """
        try:
            # Validate input data
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                raise ValidationError(f"Missing required columns: {required_columns}")
            
            # Create feature set
            features = df[required_columns].copy()
            
            # Add technical indicators if not present
            if 'SMA_20' not in features.columns:
                features['SMA_20'] = features['Close'].rolling(window=20).mean()
            if 'SMA_50' not in features.columns:
                features['SMA_50'] = features['Close'].rolling(window=50).mean()
            if 'RSI' not in features.columns:
                features['RSI'] = self._calculate_rsi(features['Close'])
                
            # Drop any rows with NaN values
            features = features.dropna()
            
            # Scale features
            scaled_data = {}
            for column in features.columns:
                if column not in self.feature_scalers:
                    self.feature_scalers[column] = MinMaxScaler()
                
                values = features[column].values.reshape(-1, 1)
                scaled_data[column] = self.feature_scalers[column].fit_transform(values)
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(features)):
                # Input sequence
                sequence = []
                for column in features.columns:
                    sequence.append(scaled_data[column][i-sequence_length:i])
                X.append(np.column_stack(sequence))
                
                # Target value (next day's closing price)
                target = features[target_column].iloc[i]
                y.append(target)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
            
    def prepare_prediction_data(self,
                              df: pd.DataFrame,
                              sequence_length: int = Config.MODEL_SEQUENCE_LENGTH) -> np.ndarray:
        """
        Prepare data for making predictions
        
        Args:
            df: Input DataFrame
            sequence_length: Length of input sequences
            
        Returns:
            np.ndarray: Prepared input data
        """
        try:
            if len(df) < sequence_length:
                raise ValidationError(f"Not enough data points. Need at least {sequence_length}")
                
            # Get latest sequence
            latest_data = df.tail(sequence_length).copy()
            
            # Scale features
            scaled_sequence = []
            for column in latest_data.columns:
                if column in self.feature_scalers:
                    scaler = self.feature_scalers[column]
                    values = latest_data[column].values.reshape(-1, 1)
                    scaled_sequence.append(scaler.transform(values))
                    
            # Stack features
            X = np.column_stack(scaled_sequence)
            return np.array([X])
            
        except Exception as e:
            logger.error(f"Error preparing prediction data: {str(e)}")
            raise
            
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Convert scaled predictions back to original scale
        
        Args:
            predictions: Scaled predictions
            
        Returns:
            np.ndarray: Original scale predictions
        """
        try:
            if 'Close' not in self.feature_scalers:
                raise ValidationError("Price scaler not fitted")
                
            # Reshape if needed
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
                
            return self.feature_scalers['Close'].inverse_transform(predictions)
            
        except Exception as e:
            logger.error(f"Error inverse transforming predictions: {str(e)}")
            raise
            
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def split_train_val_test(self, 
                          X: np.ndarray, 
                          y: np.ndarray,
                          val_size: float = 0.15,
                          test_size: float = 0.15) -> Tuple[np.ndarray, ...]:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Input features
            y: Target values
            val_size: Validation set size
            test_size: Test set size
            
        Returns:
            Tuple[np.ndarray, ...]: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            # Calculate split indices
            n_samples = len(X)
            test_idx = int(n_samples * (1 - test_size))
            val_idx = int(test_idx * (1 - val_size))
            
            # Split data
            X_train = X[:val_idx]
            X_val = X[val_idx:test_idx]
            X_test = X[test_idx:]
            
            y_train = y[:val_idx]
            y_val = y[val_idx:test_idx]
            y_test = y[test_idx:]
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
            
    def save_scalers(self, path: str):
        """Save fitted scalers"""
        try:
            import joblib
            scaler_dict = {
                'feature_scalers': self.feature_scalers
            }
            joblib.dump(scaler_dict, path)
            
        except Exception as e:
            logger.error(f"Error saving scalers: {str(e)}")
            raise
            
    def load_scalers(self, path: str):
        """Load saved scalers"""
        try:
            import joblib
            scaler_dict = joblib.load(path)
            self.feature_scalers = scaler_dict['feature_scalers']
            
        except Exception as e:
            logger.error(f"Error loading scalers: {str(e)}")
            raise