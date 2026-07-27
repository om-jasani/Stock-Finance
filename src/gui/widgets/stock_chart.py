from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QComboBox, QFrame, QLabel)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import pyqtSlot, QUrl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from ...utils.data_fetcher import DataFetcher
import plotly.io as pio
import tempfile
import os

class StockChartWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.data_fetcher = DataFetcher()
        self.current_symbol = None
        self.temp_dir = tempfile.mkdtemp()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Control panel
        control_panel = QFrame()
        control_panel.setObjectName("control-panel")
        control_layout = QHBoxLayout(control_panel)
        
        # Time period selector
        self.period_selector = QComboBox()
        self.period_selector.addItems(['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'])
        self.period_selector.setCurrentText('1d')
        self.period_selector.currentTextChanged.connect(self.update_chart)
        control_layout.addWidget(QLabel("Time Period:"))
        control_layout.addWidget(self.period_selector)
        
        # Interval selector
        self.interval_selector = QComboBox()
        self.interval_selector.addItems(['1m', '2m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo'])
        self.interval_selector.setCurrentText('5m')
        self.interval_selector.currentTextChanged.connect(self.update_chart)
        control_layout.addWidget(QLabel("Interval:"))
        control_layout.addWidget(self.interval_selector)
        
        # Indicator toggles
        self.sma_button = QPushButton("SMA")
        self.sma_button.setCheckable(True)
        self.sma_button.setChecked(True)
        self.sma_button.clicked.connect(self.update_chart)
        control_layout.addWidget(self.sma_button)
        
        self.volume_button = QPushButton("Volume")
        self.volume_button.setCheckable(True)
        self.volume_button.setChecked(True)
        self.volume_button.clicked.connect(self.update_chart)
        control_layout.addWidget(self.volume_button)
        
        self.rsi_button = QPushButton("RSI")
        self.rsi_button.setCheckable(True)
        self.rsi_button.clicked.connect(self.update_chart)
        control_layout.addWidget(self.rsi_button)
        
        layout.addWidget(control_panel)
        
        # Chart view
        self.web_view = QWebEngineView()
        self.web_view.setMinimumHeight(500)
        layout.addWidget(self.web_view)
        
        # Info panel
        self.info_panel = QFrame()
        self.info_panel.setObjectName("info-panel")
        info_layout = QHBoxLayout(self.info_panel)
        
        # Add info labels
        self.price_label = QLabel()
        self.change_label = QLabel()
        self.volume_label = QLabel()
        
        info_layout.addWidget(self.price_label)
        info_layout.addWidget(self.change_label)
        info_layout.addWidget(self.volume_label)
        
        layout.addWidget(self.info_panel)
        
    @pyqtSlot(str)
    def update_symbol(self, symbol: str):
        """Update the chart with new symbol"""
        self.current_symbol = symbol
        self.update_chart()
        
    def update_chart(self):
        """Update the chart with current settings"""
        if not self.current_symbol:
            return
            
        # Fetch data
        df = self.data_fetcher.get_stock_data(
            self.current_symbol,
            period=self.period_selector.currentText(),
            interval=self.interval_selector.currentText()
        )
        
        if df is None or df.empty:
            return
            
        # Create figure with secondary y-axis
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                          vertical_spacing=0.05,
                          row_heights=[0.6, 0.2, 0.2])
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ), row=1, col=1)
        
        # Add SMA if enabled
        if self.sma_button.isChecked():
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['SMA_20'],
                name='SMA 20',
                line=dict(color='orange')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['SMA_50'],
                name='SMA 50',
                line=dict(color='blue')
            ), row=1, col=1)
        
        # Add volume if enabled
        if self.volume_button.isChecked():
            colors = ['red' if row['Open'] > row['Close'] else 'green'
                     for index, row in df.iterrows()]
            
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors
            ), row=2, col=1)
        
        # Add RSI if enabled
        if self.rsi_button.isChecked():
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='purple')
            ), row=3, col=1)
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{self.current_symbol} Stock Price',
            yaxis_title='Stock Price',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Save as HTML and load in QWebEngineView
        html_path = os.path.join(self.temp_dir, 'temp_chart.html')
        fig.write_html(html_path)
        self.web_view.setUrl(QUrl.fromLocalFile(html_path))
        
        # Update info panel
        last_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = last_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        self.price_label.setText(f"Price: ₹{last_price:.2f}")
        self.change_label.setText(f"Change: {price_change:+.2f} ({price_change_pct:+.2f}%)")
        self.volume_label.setText(f"Volume: {df['Volume'].iloc[-1]:,.0f}")
        
    def closeEvent(self, event):
        """Clean up temporary files on close"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass
        super().closeEvent(event)