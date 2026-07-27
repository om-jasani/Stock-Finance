"""Main application entry point"""
import sys
import os
import logging
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from .gui.main_window import MainWindow
from .utils.logger import setup_logger
from .utils.config import Config
from .utils.model_manager import ModelManager

def initialize_app():
    """Initialize application components"""
    try:
        # Create necessary directories
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        
        # Initialize logging
        setup_logger('stock_analysis')
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Validate configuration
        if not Config.validate():
            raise RuntimeError("Invalid configuration")
            
        return True
        
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        return False

def main():
    """Main entry point"""
    try:
        # Initialize components
        if not initialize_app():
            sys.exit(1)
            
        # Create application
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # Load stylesheet
        style_file = os.path.join(os.path.dirname(__file__), 'gui/styles/main.qss')
        if os.path.exists(style_file):
            with open(style_file, 'r') as f:
                app.setStyleSheet(f.read())
        
        # Create and show main window
        window = MainWindow()
        window.show()
        
        # Run event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()