"""
Stock Market Analysis & Prediction Application
"""

import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Create necessary directories if they don't exist
DIRS = ['data', 'models', 'logs']
for dir_name in DIRS:
    os.makedirs(dir_name, exist_ok=True)