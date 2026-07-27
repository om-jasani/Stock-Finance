"""Stock price prediction model"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import logging
from typing import Tuple, Optional, Dict, Any
import joblib
from .exceptions import ModelError, ModelTrainingError, ModelPredictionError, ValidationError
from .validation import validate_model_input
from .logger import logger
from .config import Config
from .data_pipeline import DataPipeline

class StockPredictor:
    """LSTM-based stock price prediction model"""
    
    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """Initialize the predictor"""
        self.model = None
        self.data_pipeline = DataPipeline()
        self.model_path = model_path or os.path.join(Config.MODELS_DIR, 'stock_predictor.keras')
        self.scaler_path = scaler_path or os.path.join(Config.MODELS_DIR, 'scalers.pkl')
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build LSTM model architecture"""
        try:
            self.model = Sequential([
                # First LSTM layer
                LSTM(units=100,
                     return_sequences=True,
                     input_shape=input_shape,
                     kernel_initializer='he_normal'),
                Dropout(0.3),
                
                # Second LSTM layer
                LSTM(units=80,
                     return_sequences=True,
                     kernel_initializer='he_normal'),
                Dropout(0.3),
                
                # Third LSTM layer
                LSTM(units=50,
                     kernel_initializer='he_normal'),
                Dropout(0.3),
                
                # Dense layers
                Dense(units=32, activation='relu'),
                Dropout(0.2),
                Dense(units=16, activation='relu'),
                
                # Output layer
                Dense(units=1)
            ])
            
            # Compile model
            optimizer = Adam(
                learning_rate=Config.MODEL_LEARNING_RATE,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )
            
            self.model.compile(
                optimizer=optimizer,
                loss='huber',  # More robust to outliers
                metrics=['mae', 'mse']
            )
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise ModelError(f"Failed to build model: {str(e)}")
            
    def train(self,
             df: pd.DataFrame,
             validation_split: float = 0.2,
             epochs: int = Config.MODEL_EPOCHS,
             batch_size: int = Config.MODEL_BATCH_SIZE) -> Dict[str, Any]:
        """Train the model"""
        try:
            # Prepare data
            X, y = self.data_pipeline.prepare_training_data(
                df,
                sequence_length=Config.MODEL_SEQUENCE_LENGTH
            )
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = \
                self.data_pipeline.split_train_val_test(X, y)
            
            # Build model if not already built
            if self.model is None:
                self.build_model(input_shape=(X.shape[1], X.shape[2]))
            
            # Create callbacks
            callbacks = [
                # Early stopping
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    mode='min'
                ),
                
                # Model checkpoint
                ModelCheckpoint(
                    self.model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=1
                ),
                
                # Reduce learning rate when plateau
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save scalers
            self.data_pipeline.save_scalers(self.scaler_path)
            
            # Calculate final metrics
            test_loss = self.model.evaluate(X_test, y_test, verbose=0)
            
            return {
                'history': history.history,
                'test_loss': test_loss,
                'final_epoch': len(history.history['loss'])
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise ModelTrainingError(f"Failed to train model: {str(e)}")
            
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        try:
            if self.model is None:
                if not self.load_model():
                    raise ModelError("No trained model available")
                    
            # Prepare data
            X = self.data_pipeline.prepare_prediction_data(
                df,
                sequence_length=Config.MODEL_SEQUENCE_LENGTH
            )
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Inverse transform predictions
            return self.data_pipeline.inverse_transform_predictions(predictions)
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise ModelPredictionError(f"Failed to make predictions: {str(e)}")
            
    def load_model(self) -> bool:
        """Load saved model and scalers"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = load_model(self.model_path)
                self.data_pipeline.load_scalers(self.scaler_path)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ModelError(f"Failed to load model: {str(e)}")
            
    def predict_future(self,
                     df: pd.DataFrame,
                     days: int = 30,
                     confidence_interval: bool = True) -> pd.DataFrame:
        """Predict future stock prices"""
        try:
            if self.model is None:
                if not self.load_model():
                    raise ModelError("No trained model available")
                    
            # Get latest sequence
            X = self.data_pipeline.prepare_prediction_data(df)
            
            predictions = []
            confidence_intervals = []
            
            # Generate predictions with Monte Carlo simulation
            n_simulations = 100 if confidence_interval else 1
            
            for _ in range(days):
                day_predictions = []
                
                for _ in range(n_simulations):
                    pred = self.model.predict(X, verbose=0)
                    day_predictions.append(pred[0, 0])
                    
                # Calculate statistics
                mean_pred = np.mean(day_predictions)
                std_pred = np.std(day_predictions)
                
                predictions.append(mean_pred)
                if confidence_interval:
                    confidence_intervals.append({
                        'lower': mean_pred - 2 * std_pred,
                        'upper': mean_pred + 2 * std_pred
                    })
                    
                # Update sequence for next prediction
                if len(X.shape) == 3:  # Check if X is 3D (batch, sequence, features)
                    X = np.roll(X, -1, axis=1)
                    X[0, -1, 0] = mean_pred  # Update closing price
                else:
                    raise ValidationError("Unexpected input shape")
                    
            # Create future dates
            last_date = pd.to_datetime(df.index[-1])
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
            
            # Create predictions DataFrame
            predictions = self.data_pipeline.inverse_transform_predictions(
                np.array(predictions).reshape(-1, 1)
            )
            
            results_df = pd.DataFrame(
                index=future_dates,
                data={'Predicted_Close': predictions.flatten()}
            )
            
            if confidence_interval:
                lower_bounds = np.array([ci['lower'] for ci in confidence_intervals])
                upper_bounds = np.array([ci['upper'] for ci in confidence_intervals])
                
                results_df['Lower_Bound'] = self.data_pipeline.inverse_transform_predictions(
                    lower_bounds.reshape(-1, 1)
                ).flatten()
                
                results_df['Upper_Bound'] = self.data_pipeline.inverse_transform_predictions(
                    upper_bounds.reshape(-1, 1)
                ).flatten()
                
            return results_df
            
        except Exception as e:
            logger.error(f"Error predicting future prices: {str(e)}")
            raise ModelPredictionError(f"Failed to predict future prices: {str(e)}")
            
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            # Prepare test data
            X, y = self.data_pipeline.prepare_training_data(df)
            
            # Make predictions
            y_pred = self.predict(df)
            y_true = self.data_pipeline.inverse_transform_predictions(y.reshape(-1, 1))
            
            # Calculate metrics
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Calculate directional accuracy
            direction_correct = np.sum(np.sign(np.diff(y_true.flatten())) == 
                                    np.sign(np.diff(y_pred.flatten())))
            direction_accuracy = direction_correct / (len(y_true) - 1) * 100
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'direction_accuracy': direction_accuracy
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise ModelError(f"Failed to evaluate model: {str(e)}")