"""Custom exceptions for the application"""

class StockAnalysisError(Exception):
    """Base exception for the application"""
    pass

class DataFetchError(StockAnalysisError):
    """Raised when there's an error fetching stock data"""
    pass

class ModelError(StockAnalysisError):
    """Base class for model-related errors"""
    pass

class ModelTrainingError(ModelError):
    """Raised when there's an error during model training"""
    pass

class ModelPredictionError(ModelError):
    """Raised when there's an error during prediction"""
    pass

class ValidationError(StockAnalysisError):
    """Raised when there's a validation error"""
    pass

class ConfigurationError(StockAnalysisError):
    """Raised when there's a configuration error"""
    pass

class DatabaseError(StockAnalysisError):
    """Raised when there's a database error"""
    pass

class UIError(StockAnalysisError):
    """Raised when there's a UI-related error"""
    pass

class NetworkError(StockAnalysisError):
    """Raised when there's a network-related error"""
    pass