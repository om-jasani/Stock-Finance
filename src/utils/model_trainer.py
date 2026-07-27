"""Stock price prediction model (PyTorch)

Trains an LSTM price predictor. Device selection is automatic - CUDA is
used when available (falling back to Apple Silicon MPS, then CPU), with
mixed precision, cuDNN autotuning, and multi-GPU DataParallel enabled
whenever the hardware supports them, so the same code runs unmodified
from a laptop GPU up to a multi-GPU training server.
"""
import os
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .exceptions import ModelError, ModelTrainingError, ModelPredictionError, ValidationError
from .logger import logger
from .config import Config
from .data_pipeline import DataPipeline


def select_device() -> torch.device:
    """Pick the best available compute device and tune backend flags for it"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        logger.info("No GPU detected, using CPU")
    return device


class _LSTMNet(nn.Module):
    """Stacked LSTM regressor. Defaults match the original Keras architecture
    (100/80/50 LSTM units, 0.3/0.3/0.3/0.2 dropout); hidden_sizes/dropout are
    parameterized so hyperparameter search can tune them."""

    def __init__(self, n_features: int, hidden_sizes: Tuple[int, int, int] = (100, 80, 50),
                fc_sizes: Tuple[int, int] = (32, 16), dropout: float = 0.3):
        super().__init__()
        h1, h2, h3 = hidden_sizes
        fc1_size, fc2_size = fc_sizes
        self.lstm1 = nn.LSTM(n_features, h1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(h2, h3, batch_first=True)
        self.drop3 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(h3, fc1_size)
        self.drop4 = nn.Dropout(dropout * 0.67)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        x, _ = self.lstm3(x)
        x = self.drop3(x[:, -1, :])  # last timestep only, like Keras return_sequences=False
        x = self.relu(self.fc1(x))
        x = self.drop4(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class StockPredictor:
    """LSTM-based stock price prediction model"""

    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None,
                hidden_sizes: Tuple[int, int, int] = (100, 80, 50),
                fc_sizes: Tuple[int, int] = (32, 16), dropout: float = 0.3,
                sequence_length: int = Config.MODEL_SEQUENCE_LENGTH,
                learning_rate: float = Config.MODEL_LEARNING_RATE):
        """Initialize the predictor.

        hidden_sizes/fc_sizes/dropout/sequence_length/learning_rate expose the
        architecture and training hyperparameters for hyperparameter search
        (see hyperparameter_search.py); defaults reproduce the original
        fixed architecture/config values. sequence_length is persisted
        alongside the scalers so a loaded model always uses the same
        windowing it was trained with.
        """
        self.model: Optional[nn.Module] = None
        self.n_features: Optional[int] = None
        self.data_pipeline = DataPipeline()
        self.model_path = model_path or os.path.join(Config.MODELS_DIR, 'stock_predictor.pt')
        self.scaler_path = scaler_path or os.path.join(Config.MODELS_DIR, 'scalers.pkl')
        self.hidden_sizes = hidden_sizes
        self.fc_sizes = fc_sizes
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.history: Optional[Dict[str, Any]] = None
        self.device = select_device()

    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build LSTM model architecture. input_shape = (sequence_length, n_features)"""
        try:
            self.n_features = input_shape[1]
            model = _LSTMNet(self.n_features, self.hidden_sizes, self.fc_sizes, self.dropout).to(self.device)

            if torch.cuda.device_count() > 1:
                logger.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
                model = nn.DataParallel(model)

            if self.device.type == 'cuda':
                try:
                    model = torch.compile(model)
                except Exception as e:
                    logger.warning(f"torch.compile unavailable, using eager mode: {e}")

            self.model = model

        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise ModelError(f"Failed to build model: {str(e)}")

    def _unwrap(self) -> nn.Module:
        """Get the underlying module, unwrapping DataParallel/compile wrappers for state_dict I/O.

        torch.compile() wraps the module in an OptimizedModule whose
        state_dict keys are prefixed with '_orig_mod.' - without stripping
        that here, a state_dict saved from a compiled model can't be loaded
        into a freshly-built (not yet compiled) or exported module.
        """
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        return model

    def _gradient_accumulation_steps(self) -> int:
        """Automatically accumulate gradients on memory-constrained GPUs to
        keep the configured batch size's optimization behavior without
        exceeding VRAM."""
        if self.device.type != 'cuda':
            return 1
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return 2 if total_memory_gb < 6 else 1

    def train(self,
             df: pd.DataFrame,
             epochs: int = Config.MODEL_EPOCHS,
             batch_size: int = Config.MODEL_BATCH_SIZE,
             patience: int = 15) -> Dict[str, Any]:
        """Train the model with early stopping, LR scheduling, and best-checkpoint saving"""
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_pipeline.prepare_training_data(
                df, sequence_length=self.sequence_length
            )

            if self.model is None:
                self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

            accumulation_steps = self._gradient_accumulation_steps()
            micro_batch_size = max(1, batch_size // accumulation_steps)

            use_pin_memory = self.device.type == 'cuda'
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
                batch_size=micro_batch_size, shuffle=True, pin_memory=use_pin_memory
            )
            val_x = torch.from_numpy(X_val).float().to(self.device, non_blocking=True)
            val_y = torch.from_numpy(y_val).float().to(self.device, non_blocking=True)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
            )
            loss_fn = nn.HuberLoss()
            use_amp = self.device.type == 'cuda'
            grad_scaler = torch.amp.GradScaler(enabled=use_amp)

            best_val_loss = float('inf')
            best_state = None
            epochs_without_improvement = 0
            history = {'loss': [], 'val_loss': []}
            final_epoch = 0

            for epoch in range(epochs):
                self.model.train()
                optimizer.zero_grad()
                running_loss = 0.0
                n_batches = 0

                for step, (xb, yb) in enumerate(train_loader):
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)

                    with torch.autocast(device_type='cuda', enabled=use_amp):
                        preds = self.model(xb).squeeze(-1)
                        loss = loss_fn(preds, yb) / accumulation_steps

                    grad_scaler.scale(loss).backward()

                    if (step + 1) % accumulation_steps == 0:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        optimizer.zero_grad()

                    running_loss += loss.item() * accumulation_steps
                    n_batches += 1

                train_loss = running_loss / max(1, n_batches)

                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(val_x).squeeze(-1)
                    val_loss = loss_fn(val_preds, val_y).item()

                scheduler.step(val_loss)
                history['loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                final_epoch = epoch + 1

                logger.info(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().clone() for k, v in self._unwrap().state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

            if best_state is not None:
                self._unwrap().load_state_dict(best_state)

            self.model.eval()
            with torch.no_grad():
                test_x = torch.from_numpy(X_test).float().to(self.device)
                test_y = torch.from_numpy(y_test).float().to(self.device)
                test_loss = loss_fn(self.model(test_x).squeeze(-1), test_y).item()

            self.history = history
            return {
                'history': history,
                'test_loss': test_loss,
                'final_epoch': final_epoch
            }

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise ModelTrainingError(f"Failed to train model: {str(e)}")

    def save(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None) -> None:
        """Persist the trained model weights and fitted scalers"""
        if self.model is None:
            raise ModelError("No model to save; train or build a model first")

        model_path = model_path or self.model_path
        scaler_path = scaler_path or self.scaler_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self._unwrap().state_dict(), model_path)
        self.data_pipeline.save_scalers(scaler_path, extra={
            'sequence_length': self.sequence_length,
            'hidden_sizes': self.hidden_sizes,
            'fc_sizes': self.fc_sizes,
            'dropout': self.dropout,
        })
        self.model_path, self.scaler_path = model_path, scaler_path

    def _export_ready_model(self) -> nn.Module:
        """A fresh, uncompiled CPU copy of the current weights - safe to
        trace/export regardless of whether self.model is wrapped in
        DataParallel or torch.compile's OptimizedModule."""
        if self.model is None:
            raise ModelError("No model to export; train or load a model first")
        export_model = _LSTMNet(self.n_features, self.hidden_sizes, self.fc_sizes, self.dropout)
        export_model.load_state_dict(self._unwrap().state_dict())
        export_model.eval()
        return export_model

    def export_torchscript(self, path: str) -> None:
        """Export a TorchScript module for serving independent of this training code/environment"""
        export_model = self._export_ready_model()
        example_input = torch.randn(1, self.sequence_length, self.n_features)
        traced = torch.jit.trace(export_model, example_input)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        traced.save(path)
        logger.info(f"Exported TorchScript model to {path}")

    def export_onnx(self, path: str) -> None:
        """Export an ONNX model for serving independent of this training code/environment"""
        export_model = self._export_ready_model()
        example_input = torch.randn(1, self.sequence_length, self.n_features)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # torch.onnx's exporter prints unicode status glyphs; on a Windows
        # console with a non-UTF-8 code page (the default), that print can
        # itself raise UnicodeEncodeError after the export already
        # succeeded - report success from the file's existence, not just a
        # clean return, so that doesn't look like a real failure.
        try:
            torch.onnx.export(
                export_model, example_input, path,
                input_names=['sequence'], output_names=['predicted_close'],
                dynamic_axes={'sequence': {0: 'batch_size'}, 'predicted_close': {0: 'batch_size'}}
            )
        except UnicodeEncodeError:
            if not os.path.exists(path):
                raise
        logger.info(f"Exported ONNX model to {path}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        try:
            if self.model is None:
                if not self.load_model():
                    raise ModelError("No trained model available")

            X = self.data_pipeline.prepare_prediction_data(df, sequence_length=self.sequence_length)

            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.from_numpy(X).float().to(self.device)
                predictions = self.model(x_tensor).cpu().numpy()

            return self.data_pipeline.inverse_transform_predictions(predictions)

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise ModelPredictionError(f"Failed to make predictions: {str(e)}")

    def load_model(self) -> bool:
        """Load saved model and scalers"""
        try:
            if not (os.path.exists(self.model_path) and os.path.exists(self.scaler_path)):
                return False

            saved = self.data_pipeline.load_scalers(self.scaler_path)
            self.sequence_length = saved.get('sequence_length', self.sequence_length)
            self.hidden_sizes = tuple(saved.get('hidden_sizes', self.hidden_sizes))
            self.fc_sizes = tuple(saved.get('fc_sizes', self.fc_sizes))
            self.dropout = saved.get('dropout', self.dropout)

            n_features = len(self.data_pipeline.feature_columns)
            self.build_model(input_shape=(self.sequence_length, n_features))
            state_dict = torch.load(self.model_path, map_location=self.device)
            self._unwrap().load_state_dict(state_dict)
            self.model.eval()
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ModelError(f"Failed to load model: {str(e)}")

    def predict_future(self,
                     df: pd.DataFrame,
                     days: int = 30,
                     confidence_interval: bool = True) -> pd.DataFrame:
        """Predict future stock prices using batched Monte Carlo dropout for uncertainty"""
        try:
            if self.model is None:
                if not self.load_model():
                    raise ModelError("No trained model available")

            X = self.data_pipeline.prepare_prediction_data(df, sequence_length=self.sequence_length)
            x_tensor = torch.from_numpy(X).float().to(self.device)

            n_simulations = 100 if confidence_interval else 1
            predictions = []
            confidence_intervals = []

            # Dropout must stay active for MC sampling, but no gradients are needed
            self.model.train()
            with torch.no_grad():
                for _ in range(days):
                    # One batched forward pass covers all MC simulations for this day,
                    # instead of looping n_simulations sequential single-sample calls.
                    batch = x_tensor.repeat(n_simulations, 1, 1)
                    day_predictions = self.model(batch).squeeze(-1).cpu().numpy()

                    mean_pred = float(day_predictions.mean())
                    std_pred = float(day_predictions.std())

                    predictions.append(mean_pred)
                    if confidence_interval:
                        confidence_intervals.append({
                            'lower': mean_pred - 2 * std_pred,
                            'upper': mean_pred + 2 * std_pred
                        })

                    # Roll the sequence forward, appending the predicted close
                    x_tensor = torch.roll(x_tensor, shifts=-1, dims=1)
                    x_tensor[0, -1, 0] = mean_pred
            self.model.eval()

            last_date = pd.to_datetime(df.index[-1])
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)

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
        """Evaluate model performance on a held-out DataFrame"""
        try:
            if self.model is None:
                if not self.load_model():
                    raise ModelError("No trained model available")

            _, _, X_test, _, _, y_test = self.data_pipeline.prepare_training_data(
                df, sequence_length=self.sequence_length
            )

            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.from_numpy(X_test).float().to(self.device)
                y_pred_scaled = self.model(x_tensor).cpu().numpy()

            y_pred = self.data_pipeline.inverse_transform_predictions(y_pred_scaled)
            y_true = self.data_pipeline.inverse_transform_predictions(y_test.reshape(-1, 1))

            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            direction_correct = np.sum(np.sign(np.diff(y_true.flatten())) ==
                                    np.sign(np.diff(y_pred.flatten())))
            direction_accuracy = direction_correct / (len(y_true) - 1) * 100 if len(y_true) > 1 else 0.0

            return {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'direction_accuracy': float(direction_accuracy)
            }

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise ModelError(f"Failed to evaluate model: {str(e)}")
