#!/usr/bin/env python3
"""
Launch script for Stock Market Analysis & Prediction Application
"""

import sys
import os
import logging
from src.main import main

if __name__ == '__main__':
    try:
        # Add source directory to Python path
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('app.log'),
                logging.StreamHandler()
            ]
        )
        
        # Your code to launch the application
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)