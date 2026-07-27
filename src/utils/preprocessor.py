"""Data preprocessing utilities"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import MinMaxScaler
from .validation import validate_stock_data
from .exceptions import ValidationError
from .logger import logger
from .config import Config

class DataPreprocessor:
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.feature_scalers = {}
        
    def prepare_data(self, 
                    df: pd.DataFrame,
                    sequence_length: int = Config.MODEL_SEQUENCE_LENGTH,
                    target_column: str = 'Close',
                    feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training/prediction
        
        Args:
            df: Input DataFrame
            sequence_length: Number of time steps in each sequence
            target_column: Column to predict
            feature_columns: Additional feature columns to include
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) preprocessed data
        """
        try:
            # Validate input data
            validate_stock_data(df)
            
            # Define default feature columns if not provided
            if feature_columns is None:
                feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Ensure all required columns exist
            missing_cols = [col for col in feature_columns + [target_column] 
                          if col not in df.columns]
            if missing_cols:
                raise ValidationError(f"Missing columns: {missing_cols}")
            
            # Create feature DataFrame
            feature_data = df[feature_columns].copy()
            
            # Scale features
            scaled_features = {}
            for column in feature_columns:
                if column not in self.feature_scalers:
                    self.feature_scalers[column] = MinMaxScaler()
                    
                values = feature_data[column].values.reshape(-1, 1)
                scaled_features[column] = self.feature_scalers[column].fit_transform(values)
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(df)):
                # Get sequence of scaled features
                sequence = []
                for column in feature_columns:
                    sequence.append(scaled_features[column][i-sequence_length:i])
                X.append(np.column_stack(sequence))
                
                # Get target value
                target = df[target_column].iloc[i]
                y.append(target)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
            
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        try:
            # Validate input data
            validate_stock_data(df)
            
            # Create copy of DataFrame
            df = df.copy()
            
            # Add Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            
            # Add Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
            df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
            
            # Add RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Add MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Add Average True Range (ATR)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
            # Add Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise
            
    def inverse_transform_predictions(self, 
                                   predictions: np.ndarray,
                                   feature_name: str = 'Close') -> np.ndarray:
        """
        Inverse transform scaled predictions
        
        Args:
            predictions: Scaled predictions
            feature_name: Name of the feature to inverse transform
            
        Returns:
            np.ndarray: Original scale predictions
        """
        try:
            if feature_name not in self.feature_scalers:
                raise ValidationError(f"No scaler found for feature: {feature_name}")
                
            scaler = self.feature_scalers[feature_name]
            
            # Reshape predictions if needed
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
                
            return scaler.inverse_transform(predictions)
            
        except Exception as e:
            logger.error(f"Error inverse transforming predictions: {str(e)}")
            raise