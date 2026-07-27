"""Validation utilities for the application"""
import re
import datetime
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any
from .exceptions import ValidationError
from .logger import logger

# Ticker symbols: letters, and optionally digits/dot/dash for share classes
# and exchange suffixes (e.g. BRK.B, BF-B, RELIANCE.NS)
_SYMBOL_PATTERN = re.compile(r'^[A-Za-z0-9]+([.\-][A-Za-z0-9]+)*$')

def validate_stock_data(df: pd.DataFrame) -> bool:
    """
    Validate stock data DataFrame
    
    Args:
        df: DataFrame to validate
        
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    try:
        # Check if DataFrame is empty
        if df is None or df.empty:
            raise ValidationError("DataFrame is empty")
            
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")
            
        # Check for NaN values
        if df[required_columns].isna().any().any():
            raise ValidationError("DataFrame contains NaN values")
            
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        if (df[price_columns] < 0).any().any():
            raise ValidationError("DataFrame contains negative prices")
            
        # Check for negative volume
        if (df['Volume'] < 0).any():
            raise ValidationError("DataFrame contains negative volume")
            
        # Check for chronological order
        if not df.index.is_monotonic_increasing:
            raise ValidationError("DataFrame is not in chronological order")
            
        return True

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Data validation error: {str(e)}")
        raise ValidationError(f"Data validation failed: {str(e)}")

def validate_model_input(X: Union[np.ndarray, pd.DataFrame], 
                       sequence_length: int,
                       n_features: int) -> bool:
    """
    Validate model input data
    
    Args:
        X: Input data
        sequence_length: Expected sequence length
        n_features: Expected number of features
        
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Check dimensions
        if len(X.shape) != 3:
            raise ValidationError(f"Expected 3D input, got shape {X.shape}")
            
        if X.shape[1] != sequence_length:
            raise ValidationError(
                f"Expected sequence length {sequence_length}, got {X.shape[1]}"
            )
            
        if X.shape[2] != n_features:
            raise ValidationError(
                f"Expected {n_features} features, got {X.shape[2]}"
            )
            
        # Check for NaN values
        if np.isnan(X).any():
            raise ValidationError("Input contains NaN values")
            
        # Check for infinite values
        if np.isinf(X).any():
            raise ValidationError("Input contains infinite values")
            
        return True

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Input validation error: {str(e)}")
        raise ValidationError(f"Input validation failed: {str(e)}")

def validate_portfolio_data(data: Dict[str, Any]) -> bool:
    """
    Validate portfolio data
    
    Args:
        data: Portfolio data dictionary
        
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        required_fields = ['symbol', 'shares', 'purchase_price']
        
        # Check if dictionary is empty
        if not data:
            raise ValidationError("Portfolio data is empty")
            
        # Validate each position
        for symbol, position in data.items():
            # Check symbol format (allows share-class/exchange suffixes like BRK.B, BF-B)
            if not isinstance(symbol, str) or not _SYMBOL_PATTERN.match(symbol):
                raise ValidationError(f"Invalid symbol format: {symbol}")
                
            # Check required fields
            missing_fields = [field for field in required_fields 
                            if field not in position]
            if missing_fields:
                raise ValidationError(
                    f"Missing required fields for {symbol}: {missing_fields}"
                )
                
            # Validate numeric values
            if not isinstance(position['shares'], (int, float)) or position['shares'] <= 0:
                raise ValidationError(
                    f"Invalid shares value for {symbol}: {position['shares']}"
                )
                
            if not isinstance(position['purchase_price'], (int, float)) or position['purchase_price'] <= 0:
                raise ValidationError(
                    f"Invalid purchase price for {symbol}: {position['purchase_price']}"
                )
                
        return True

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Portfolio validation error: {str(e)}")
        raise ValidationError(f"Portfolio validation failed: {str(e)}")

def validate_date_range(start_date: pd.Timestamp, 
                       end_date: pd.Timestamp,
                       min_days: int = 1,
                       max_days: int = 365 * 5) -> bool:
    """
    Validate date range
    
    Args:
        start_date: Start date
        end_date: End date
        min_days: Minimum number of days
        max_days: Maximum number of days
        
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Check if dates are valid (accept any datetime-like, not just pd.Timestamp)
        if not isinstance(start_date, datetime.date) or not isinstance(end_date, datetime.date):
            raise ValidationError("Invalid date format")


        # Check if end date is after start date
        if end_date <= start_date:
            raise ValidationError("End date must be after start date")
            
        # Calculate date range
        days = (end_date - start_date).days
        
        # Check range limits
        if days < min_days:
            raise ValidationError(f"Date range too short: {days} days (minimum {min_days})")
            
        if days > max_days:
            raise ValidationError(f"Date range too long: {days} days (maximum {max_days})")
            
        return True

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Date range validation error: {str(e)}")
        raise ValidationError(f"Date range validation failed: {str(e)}")