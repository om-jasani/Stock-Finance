"""Stock price prediction widget"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                         QFrame, QLabel, QSpinBox, QProgressBar, QMessageBox)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import pyqtSlot, QThread, pyqtSignal, QUrl
import pandas as pd
import numpy as np
from datetime import datetime
import os
import tempfile
from ...utils.data_fetcher import DataFetcher
from ...utils.model_trainer import StockPredictor
from ...utils.model_manager import ModelManager
from ...utils.logger import logger
from ...utils.config import Config

class TrainingWorker(QThread):
    """Worker thread for model training"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, symbol: str, training_years: int = 2):
        super().__init__()
        self.symbol = symbol
        self.training_years = training_years
        self.data_fetcher = DataFetcher()
        self.model = StockPredictor()
        
    def run(self):
        try:
            # Fetch training data
            df = self.data_fetcher.get_stock_data(
                self.symbol,
                period=f"{self.training_years}y",
                interval="1d"
            )
            
            if df is None or df.empty:
                raise ValueError(f"No data available for {self.symbol}")
            
            # Train model
            training_results = self.model.train(df)
            
            # Save model through model manager
            model_manager = ModelManager()
            model_id = model_manager.save_model(
                symbol=self.symbol,
                model=self.model,
                metrics=training_results
            )
            
            # Emit results
            self.finished.emit({
                'model_id': model_id,
                'metrics': training_results
            })
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            self.error.emit(str(e))

class PredictionWidget(QWidget):
    """Widget for stock price predictions"""
    
    def __init__(self):
        super().__init__()
        self.data_fetcher = DataFetcher()
        self.model_manager = ModelManager()
        self.current_symbol = None
        self.temp_dir = tempfile.mkdtemp()
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)
        
        # Control panel
        control_panel = QFrame()
        control_panel.setObjectName("control-panel")
        control_layout = QHBoxLayout(control_panel)
        
        # Days to predict spinner
        self.days_spinner = QSpinBox()
        self.days_spinner.setRange(1, 365)
        self.days_spinner.setValue(30)
        control_layout.addWidget(QLabel("Days to Predict:"))
        control_layout.addWidget(self.days_spinner)
        
        # Training period selector
        self.period_selector = QSpinBox()
        self.period_selector.setRange(1, 10)
        self.period_selector.setValue(2)
        control_layout.addWidget(QLabel("Training Period (Years):"))
        control_layout.addWidget(self.period_selector)
        
        # Train button
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        control_layout.addWidget(self.train_button)
        
        # Predict button
        self.predict_button = QPushButton("Make Prediction")
        self.predict_button.clicked.connect(self.make_prediction)
        control_layout.addWidget(self.predict_button)
        
        layout.addWidget(control_panel)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Chart view
        self.web_view = QWebEngineView()
        self.web_view.setMinimumHeight(500)
        layout.addWidget(self.web_view)
        
        # Prediction info panel
        self.info_panel = QFrame()
        self.info_panel.setObjectName("info-panel")
        info_layout = QVBoxLayout(self.info_panel)
        
        self.prediction_label = QLabel()
        self.accuracy_label = QLabel()
        info_layout.addWidget(self.prediction_label)
        info_layout.addWidget(self.accuracy_label)
        
        layout.addWidget(self.info_panel)
        
    @pyqtSlot(str)
    def update_symbol(self, symbol: str):
        """Update current symbol"""
        self.current_symbol = symbol
        self.update_model_info()
        
    def update_model_info(self):
        """Update model information display"""
        if not self.current_symbol:
            return

        # Check for existing model
        model_id = self.model_manager.get_latest_model_id(self.current_symbol)
        if model_id:
            metrics = self.model_manager.get_model_metrics(model_id)
            if metrics:
                self.accuracy_label.setText(
                    f"Model Accuracy: {metrics.get('direction_accuracy', 0):.1f}%\n"
                    f"MAPE: {metrics.get('mape', 0):.2f}%"
                )
        else:
            self.accuracy_label.setText("No trained model available")
        
    def train_model(self):
        """Train prediction model"""
        if not self.current_symbol:
            QMessageBox.warning(self, "Error", "No stock symbol selected")
            return
            
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.train_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        
        # Start training in thread
        self.training_thread = TrainingWorker(
            self.current_symbol,
            self.period_selector.value()
        )
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.error.connect(self.on_training_error)
        self.training_thread.start()
        
    def on_training_finished(self, results: dict):
        """Handle training completion"""
        self.progress_bar.setVisible(False)
        self.train_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        
        metrics = results['metrics']
        test_loss = metrics.get('test_loss', 0)
        final_loss = test_loss[0] if isinstance(test_loss, (list, tuple)) else test_loss
        self.accuracy_label.setText(
            f"Training completed successfully\n"
            f"Final loss: {final_loss:.4f}"
        )
        
        # Make prediction with new model
        self.make_prediction()
        
    def on_training_error(self, error_msg: str):
        """Handle training error"""
        self.progress_bar.setVisible(False)
        self.train_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.accuracy_label.setText(f"Training error: {error_msg}")
        
    def make_prediction(self):
        """Make price predictions"""
        if not self.current_symbol:
            QMessageBox.warning(self, "Error", "No stock symbol selected")
            return
            
        try:
            # Get model
            model = self.model_manager.get_latest_model(self.current_symbol)
            if model is None:
                QMessageBox.warning(self, "Error", "No trained model available")
                return
            
            # Fetch recent data
            df = self.data_fetcher.get_stock_data(
                self.current_symbol,
                period="60d",
                interval="1d"
            )
            
            if df is None or df.empty:
                QMessageBox.warning(self, "Error", "Failed to fetch stock data")
                return
            
            # Make predictions
            predictions_df = model.predict_future(
                df,
                days=self.days_spinner.value(),
                confidence_interval=True
            )
            
            # Visualize predictions
            self.plot_predictions(df, predictions_df)
            
            # Update info
            last_price = df['Close'].iloc[-1]
            final_prediction = predictions_df['Predicted_Close'].iloc[-1]
            price_change = final_prediction - last_price
            price_change_pct = (price_change / last_price) * 100
            
            self.prediction_label.setText(
                f"Current Price: ₹{last_price:.2f}\n"
                f"Predicted Price ({self.days_spinner.value()} days): "
                f"₹{final_prediction:.2f} ({price_change_pct:+.2f}%)"
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to make prediction: {str(e)}")
        
    def plot_predictions(self, historical_df: pd.DataFrame, predictions_df: pd.DataFrame):
        """Plot predictions"""
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_df.index,
                y=historical_df['Close'],
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=predictions_df.index,
                y=predictions_df['Predicted_Close'],
                name='Predicted',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence intervals
            if 'Lower_Bound' in predictions_df.columns and 'Upper_Bound' in predictions_df.columns:
                fig.add_trace(go.Scatter(
                    x=predictions_df.index,
                    y=predictions_df['Upper_Bound'],
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=predictions_df.index,
                    y=predictions_df['Lower_Bound'],
                    name='Confidence Interval',
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(width=0)
                ))
            
            # Update layout
            fig.update_layout(
                title=f'{self.current_symbol} Price Prediction',
                yaxis_title='Stock Price',
                template='plotly_dark',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=600
            )
            
            # Save and display
            html_path = os.path.join(self.temp_dir, 'prediction_chart.html')
            fig.write_html(html_path)
            self.web_view.setUrl(QUrl.fromLocalFile(html_path))
            
        except Exception as e:
            logger.error(f"Error plotting predictions: {str(e)}")
            
    def closeEvent(self, event):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass
        super().closeEvent(event)