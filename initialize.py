"""Initialize project environment"""
import os
import sys
import logging
from pathlib import Path

def initialize_project():
    """Initialize project structure and environment"""
    try:
        # Create necessary directories
        directories = [
            'data',
            'data/cache',
            'logs',
            'models',
            'config'
        ]
        
        base_dir = Path(__file__).parent
        for dir_name in directories:
            dir_path = base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Create default .env if not exists
        env_file = base_dir / '.env'
        if not env_file.exists():
            with open(env_file, 'w') as f:
                f.write("""# API Keys and Configuration
DEBUG_MODE=False
LOG_LEVEL=INFO
DATA_CACHE_MINUTES=15
MAX_PREDICTION_DAYS=365""")
                
        print("Project initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing project: {str(e)}")
        return False

if __name__ == '__main__':
    initialize_project()