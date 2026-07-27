from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFrame, QLabel, QComboBox)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import pyqtSlot, QUrl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from ...utils.data_fetcher import DataFetcher
import tempfile
import os

class AnalysisWidget(QWidget):
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
        
        # Period selector
        self.period_selector = QComboBox()
        self.period_selector.addItems(['1mo', '3mo', '6mo', '1y', '2y', '5y'])
        self.period_selector.setCurrentText('1y')
        self.period_selector.currentTextChanged.connect(self.update_analysis)
        control_layout.addWidget(QLabel("Time Period:"))
        control_layout.addWidget(self.period_selector)
        
        # Technical indicators
        indicators = [
            ("Moving Averages", "ma_button"),
            ("RSI", "rsi_button"),
            ("MACD", "macd_button"),
            ("Bollinger Bands", "bb_button"),
            ("Volume Profile", "volume_button")
        ]
        
        for label, attr_name in indicators:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.clicked.connect(self.update_analysis)
            setattr(self, attr_name, btn)
            control_layout.addWidget(btn)
            
        layout.addWidget(control_panel)
        
        # Chart view
        self.web_view = QWebEngineView()
        self.web_view.setMinimumHeight(600)
        layout.addWidget(self.web_view)
        
        # Analysis panel
        analysis_panel = QFrame()
        analysis_panel.setObjectName("analysis-panel")
        analysis_layout = QVBoxLayout(analysis_panel)
        
        self.signal_label = QLabel()
        self.trend_label = QLabel()
        self.support_label = QLabel()
        self.resistance_label = QLabel()
        
        analysis_layout.addWidget(self.signal_label)
        analysis_layout.addWidget(self.trend_label)
        analysis_layout.addWidget(self.support_label)
        analysis_layout.addWidget(self.resistance_label)
        
        layout.addWidget(analysis_panel)
        
    def update_analysis(self):
        """Update technical analysis"""
        if not self.current_symbol:
            return
            
        # Fetch data
        df = self.data_fetcher.get_stock_data(
            self.current_symbol,
            period=self.period_selector.currentText(),
            interval='1d'
        )
        
        if df is None or df.empty:
            return
            
        # Create subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                          vertical_spacing=0.05,
                          row_heights=[0.6, 0.2, 0.2])
                          
        # Main price chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ), row=1, col=1)
        
        # Add Moving Averages
        if self.ma_button.isChecked():
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
            
        # Add Bollinger Bands
        if self.bb_button.isChecked():
            upper, middle, lower = self.calculate_bollinger_bands(df['Close'])
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=upper,
                name='Upper BB',
                line=dict(color='gray', dash='dash')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=lower,
                name='Lower BB',
                line=dict(color='gray', dash='dash'),
                fill='tonexty'
            ), row=1, col=1)
            
        # Add Volume
        if self.volume_button.isChecked():
            colors = ['red' if row['Open'] > row['Close'] else 'green'
                     for index, row in df.iterrows()]
            
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors
            ), row=2, col=1)
            
        # Add RSI
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
            
        # Add MACD
        if self.macd_button.isChecked():
            macd, signal, histogram = self.calculate_macd(df['Close'])
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=macd,
                name='MACD',
                line=dict(color='blue')
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=signal,
                name='Signal',
                line=dict(color='orange')
            ), row=3, col=1)
            
            fig.add_trace(go.Bar(
                x=df.index,
                y=histogram,
                name='Histogram',
                marker_color=['red' if x < 0 else 'green' for x in histogram]
            ), row=3, col=1)
            
        # Update layout
        fig.update_layout(
            title=f'{self.current_symbol} Technical Analysis',
            yaxis_title='Price',
            yaxis2_title='Volume',
            yaxis3_title='Indicators',
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
        
        # Save and display chart
        html_path = os.path.join(self.temp_dir, 'analysis_chart.html')
        fig.write_html(html_path)
        self.web_view.setUrl(QUrl.fromLocalFile(html_path))
        
        # Update analysis labels
        self.update_analysis_labels(df)
        
    def update_analysis_labels(self, df: pd.DataFrame):
        """Update technical analysis labels"""
        # Get latest values
        current_price = df['Close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd, signal, _ = self.calculate_macd(df['Close'])
        
        # Calculate support and resistance
        support_levels, resistance_levels = self.calculate_support_resistance(df)
        
        # Generate signals
        signals = []
        
        # Moving average signals
        if current_price > sma_20 and current_price > sma_50:
            trend = "Uptrend"
            if sma_20 > sma_50:
                signals.append("Strong Buy (Golden Cross)")
            else:
                signals.append("Buy")
        elif current_price < sma_20 and current_price < sma_50:
            trend = "Downtrend"
            if sma_20 < sma_50:
                signals.append("Strong Sell (Death Cross)")
            else:
                signals.append("Sell")
        else:
            trend = "Sideways"
            signals.append("Neutral")
            
        # RSI signals
        if rsi > 70:
            signals.append("Overbought")
        elif rsi < 30:
            signals.append("Oversold")
            
        # MACD signals
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            signals.append("MACD Bullish Crossover")
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            signals.append("MACD Bearish Crossover")
            
        # Update labels
        self.signal_label.setText(f"Signals: {', '.join(signals)}")
        self.trend_label.setText(f"Trend: {trend}")
        self.support_label.setText(f"Support Levels: {', '.join([f'${x:.2f}' for x in support_levels])}")
        self.resistance_label.setText(f"Resistance Levels: {', '.join([f'${x:.2f}' for x in resistance_levels])}")
        
    def closeEvent(self, event):
        """Clean up temporary files on close"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass
        super().closeEvent(event)