"""Configuration utility for the application"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
    
    # Model Configuration
    MODEL_BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE', '32'))
    MODEL_EPOCHS = int(os.getenv('MODEL_EPOCHS', '100'))
    MODEL_SEQUENCE_LENGTH = int(os.getenv('MODEL_SEQUENCE_LENGTH', '60'))
    MODEL_LEARNING_RATE = float(os.getenv('MODEL_LEARNING_RATE', '0.001'))
    
    # Application Settings
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    DATA_CACHE_MINUTES = int(os.getenv('DATA_CACHE_MINUTES', '15'))
    MAX_PREDICTION_DAYS = int(os.getenv('MAX_PREDICTION_DAYS', '365'))
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    @classmethod
    def validate(cls):
        """Validate the configuration"""
        required_dirs = [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR]
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
            
        # Validate API keys if being used
        if not cls.ALPHA_VANTAGE_API_KEY:
            print("Warning: ALPHA_VANTAGE_API_KEY not set")
            
        if not cls.FINNHUB_API_KEY:
            print("Warning: FINNHUB_API_KEY not set")
            
        return True

# Validate configuration on import
Config.validate()