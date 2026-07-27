"""Main application window"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                         QPushButton, QComboBox, QLabel, QStackedWidget,
                         QLineEdit, QCompleter)
from PyQt6.QtCore import Qt, pyqtSlot, QThread, pyqtSignal, QStringListModel
import os
import yfinance as yf
from .widget_manager import WidgetManager
from .styles.colors import ColorScheme, StyleConstants
from ..utils.logger import logger

class StockSearchThread(QThread):
    """Background thread for stock symbol search"""
    resultReady = pyqtSignal(list)
    
    def __init__(self, query):
        super().__init__()
        self.query = query
        
    def run(self):
        try:
            matches = []
            if len(self.query) >= 2:
                common_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM"]
                matches = [s for s in common_symbols if self.query.upper() in s]
            self.resultReady.emit(matches)
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            self.resultReady.emit([])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Market Analysis & Prediction")
        self.setMinimumSize(1200, 800)
        
        # Initialize widget manager
        self.widget_manager = WidgetManager()
        
        self.init_ui()
        
        # Set default stock
        self.update_stock("AAPL")
        
    def init_ui(self):
        """Initialize the user interface"""
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create header
        header_layout = self._create_header()
        main_layout.addLayout(header_layout)
        
        # Create stacked widget for different pages
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.widget_manager.get_widget('chart'))
        self.stacked_widget.addWidget(self.widget_manager.get_widget('prediction'))
        self.stacked_widget.addWidget(self.widget_manager.get_widget('portfolio'))
        self.stacked_widget.addWidget(self.widget_manager.get_widget('analysis'))
        
        # Set default page
        self.stacked_widget.setCurrentWidget(self.widget_manager.get_widget('chart'))
        
        # Apply styles
        self._apply_styles()
        
    def _create_header(self):
        """Create header with navigation buttons"""
        header_layout = QHBoxLayout()
        
        # Add logo/title
        title_label = QLabel("Stock Analysis & Prediction")
        title_label.setObjectName("header-title")
        header_layout.addWidget(title_label)
        
        # Add navigation buttons
        nav_buttons = [
            ("Charts", 'chart'),
            ("Predictions", 'prediction'),
            ("Portfolio", 'portfolio'),
            ("Analysis", 'analysis')
        ]
        
        for text, widget_name in nav_buttons:
            btn = QPushButton(text)
            btn.setObjectName("nav-button")
            btn.clicked.connect(lambda checked, w=widget_name: 
                              self.stacked_widget.setCurrentWidget(self.widget_manager.get_widget(w)))
            header_layout.addWidget(btn)
        
        # Add stock search
        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("Enter Stock Symbol...")
        self.stock_input.setMinimumWidth(150)
        self.stock_input.setObjectName("stock-input")
        self.stock_input.textChanged.connect(self.search_stocks)
        
        # Create completer
        self.completer = QCompleter([])
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.stock_input.setCompleter(self.completer)
        
        # Connect stock selection
        self.stock_input.returnPressed.connect(
            lambda: self.update_stock(self.stock_input.text()))
        
        header_layout.addWidget(self.stock_input)
        
        return header_layout
        
    def _apply_styles(self):
        """Apply styles to the window"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {ColorScheme.BACKGROUND};
                color: {ColorScheme.TEXT_PRIMARY};
            }}
            
            QLabel#header-title {{
                color: {ColorScheme.TEXT_PRIMARY};
                font-size: {StyleConstants.FONT_SIZE_HEADER}px;
                font-weight: bold;
            }}
            
            QPushButton#nav-button {{
                background-color: transparent;
                border: none;
                color: {ColorScheme.PRIMARY};
                font-weight: bold;
                padding: {StyleConstants.PADDING_NORMAL}px;
            }}
            
            QPushButton#nav-button:hover {{
                color: {ColorScheme.TEXT_PRIMARY};
            }}
            
            QLineEdit#stock-input {{
                background-color: {ColorScheme.SURFACE};
                border: 1px solid {ColorScheme.SECONDARY};
                border-radius: {StyleConstants.BORDER_RADIUS_NORMAL}px;
                color: {ColorScheme.TEXT_PRIMARY};
                padding: {StyleConstants.PADDING_NORMAL}px;
            }}
        """)
        
    def search_stocks(self, text):
        """Search for stock symbols"""
        if hasattr(self, 'search_thread'):
            self.search_thread.terminate()
            self.search_thread.wait()
            
        self.search_thread = StockSearchThread(text)
        self.search_thread.resultReady.connect(self._update_completer)
        self.search_thread.start()
        
    @pyqtSlot(list)
    def _update_completer(self, symbols):
        """Update the completer with search results"""
        model = QStringListModel()
        model.setStringList(symbols)
        self.completer.setModel(model)
        
    def update_stock(self, symbol):
        """Update all widgets with new stock symbol"""
        symbol = symbol.upper().strip()
        if not symbol:
            return
            
        try:
            # Verify stock exists
            stock = yf.Ticker(symbol)
            info = stock.info
            if info:
                # Update widgets through widget manager
                self.widget_manager.update_symbol(symbol)
                
                # Update window title
                self.setWindowTitle(f"{info.get('longName', symbol)} - Stock Analysis & Prediction")
                
        except Exception as e:
            logger.error(f"Error updating stock: {str(e)}")
            
    def closeEvent(self, event):
        """Handle application shutdown"""
        try:
            # Save portfolio data
            portfolio_widget = self.widget_manager.get_widget('portfolio')
            if portfolio_widget:
                portfolio_widget.save_portfolio()
        except:
            pass
        super().closeEvent(event)