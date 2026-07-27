"""Main application window"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                         QPushButton, QComboBox, QLabel, QStackedWidget,
                         QLineEdit, QCompleter)
from PyQt6.QtCore import Qt, pyqtSlot, QObject, pyqtSignal, QStringListModel
import os
import threading
import yfinance as yf
from .widget_manager import WidgetManager
from .styles.colors import ColorScheme, StyleConstants
from .workers import FetchWorker
from ..utils.logger import logger

# Common tickers used to seed the search completer instantly; the search
# thread also queries Yahoo Finance's lookup API for anything not in here.
_COMMON_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA",
                    "AMD", "INTC", "IBM", "NFLX", "DIS", "V", "MA", "JPM"]


class StockSearchThread(QObject):
    """Background stock symbol search.

    Deliberately a QObject driving a plain threading.Thread rather than a
    QThread subclass - see workers.py's docstring: yfinance's curl_cffi
    backend segfaults when called inside QThread.run(). PyQt signals emit
    correctly from any Python thread, so this keeps the same interface
    (start/cancel/resultReady) without that crash.
    """
    resultReady = pyqtSignal(list)

    def __init__(self, query):
        super().__init__()
        self.query = query
        self._cancelled = False
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def isRunning(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self):
        """Cooperative cancellation flag (QThread.terminate() is unsafe)"""
        self._cancelled = True

    def _run(self):
        try:
            matches = []
            query_upper = self.query.upper().strip()
            if len(query_upper) >= 2:
                matches = [s for s in _COMMON_SYMBOLS if query_upper in s]

                if self._cancelled:
                    return

                # Look up real matches via yfinance's search, so tickers
                # outside the seed list are still found.
                try:
                    results = yf.Lookup(query_upper).get_stock(count=10)
                    if results is not None and not results.empty:
                        for sym in results.index.get_level_values(0).unique():
                            sym = str(sym).upper()
                            if sym not in matches:
                                matches.append(sym)
                except Exception as e:
                    logger.debug(f"Symbol lookup unavailable for '{query_upper}': {e}")

            if not self._cancelled:
                self.resultReady.emit(matches)
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            if not self._cancelled:
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
        if hasattr(self, 'search_thread') and self.search_thread.isRunning():
            # Cooperative cancellation: QThread.terminate() can corrupt state
            # mid-call, so just let the stale thread finish and ignore its result.
            self.search_thread.cancel()
            self.search_thread.resultReady.disconnect(self._update_completer)

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

        # Verifying the ticker via yfinance is a network call - keep it off
        # the GUI thread so the window doesn't freeze while it resolves.
        self._stock_check_worker = FetchWorker(lambda: yf.Ticker(symbol).info)
        self._stock_check_worker.finished.connect(
            lambda info: self._on_stock_verified(symbol, info))
        self._stock_check_worker.error.connect(
            lambda msg: logger.error(f"Error updating stock '{symbol}': {msg}"))
        self._stock_check_worker.start()

    def _on_stock_verified(self, symbol: str, info: dict):
        """Apply the new symbol once yfinance has confirmed it exists"""
        if not info:
            logger.warning(f"No info returned for symbol '{symbol}'")
            return

        self.widget_manager.update_symbol(symbol)
        self.setWindowTitle(f"{info.get('longName', symbol)} - Stock Analysis & Prediction")

    def closeEvent(self, event):
        """Handle application shutdown"""
        try:
            # Save portfolio data
            portfolio_widget = self.widget_manager.get_widget('portfolio')
            if portfolio_widget:
                portfolio_widget.save_portfolio()
        except (OSError, IOError) as e:
            logger.error(f"Failed to save portfolio on close: {e}")
        super().closeEvent(event)