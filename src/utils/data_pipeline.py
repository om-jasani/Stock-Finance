"""Data preprocessing pipeline for model training"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler
from .logger import logger
from .exceptions import ValidationError
from .config import Config

# Feature columns used by the model, beyond raw OHLCV. Anything DataFetcher
# has already computed on the input DataFrame is reused for free; SMA_20/50
# and RSI are computed locally as a fallback so this pipeline also works on
# a bare OHLCV frame.
_OPTIONAL_INDICATOR_COLUMNS = [
    'MACD', 'Signal_Line', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'Volume_SMA'
]


class DataPipeline:
    """Handle data preprocessing for machine learning models.

    Scalers are always fit on the training slice of a chronological split
    only, and applied with .transform() everywhere else (validation, test,
    and live prediction) - this avoids look-ahead bias from letting the
    scaler see the range of future data during fitting.
    """

    def __init__(self):
        self.feature_scalers: Dict[str, MinMaxScaler] = {}
        self.feature_columns: List[str] = []

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the model's feature set from an OHLCV(+indicators) DataFrame"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValidationError(f"Missing required columns: {required_columns}")

        features = df[required_columns].copy()

        if 'SMA_20' not in df.columns:
            features['SMA_20'] = features['Close'].rolling(window=20).mean()
        else:
            features['SMA_20'] = df['SMA_20']

        if 'SMA_50' not in df.columns:
            features['SMA_50'] = features['Close'].rolling(window=50).mean()
        else:
            features['SMA_50'] = df['SMA_50']

        if 'RSI' not in df.columns:
            features['RSI'] = self._calculate_rsi(features['Close'])
        else:
            features['RSI'] = df['RSI']

        for column in _OPTIONAL_INDICATOR_COLUMNS:
            if column in df.columns:
                features[column] = df[column]

        return features.dropna()

    def prepare_training_data(self,
                            df: pd.DataFrame,
                            target_column: str = 'Close',
                            sequence_length: int = Config.MODEL_SEQUENCE_LENGTH,
                            val_size: float = 0.15,
                            test_size: float = 0.15) -> Tuple[np.ndarray, ...]:
        """
        Build leakage-safe train/validation/test sequences in one pass.

        The chronological split boundary is computed on the raw (unscaled)
        feature rows first; scalers are fit only on the training portion,
        then used to transform the entire series (transform, not fit) so
        validation/test never influence the scaler's learned range.

        Returns:
            Tuple[np.ndarray, ...]: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            features = self._build_features(df)
            self.feature_columns = list(features.columns)
            n_samples = len(features)

            min_required = sequence_length + 10
            if n_samples < min_required:
                raise ValidationError(
                    f"Not enough data to build sequences: got {n_samples} rows, "
                    f"need at least {min_required}"
                )

            # Split boundary on raw rows, before any scaling
            test_idx = int(n_samples * (1 - test_size))
            train_idx = int(test_idx * (1 - val_size))
            if train_idx <= sequence_length:
                raise ValidationError(
                    f"Training split too small ({train_idx} rows) for "
                    f"sequence_length={sequence_length}; use more history or a shorter sequence"
                )

            # Fit each feature's scaler on the training slice only
            self.feature_scalers = {}
            for column in features.columns:
                scaler = MinMaxScaler()
                scaler.fit(features[column].values[:train_idx].reshape(-1, 1))
                self.feature_scalers[column] = scaler

            # Transform the full series with the train-fitted scalers. This
            # is not leakage: the scaler's fit never saw val/test values,
            # and applying a fixed transform to later data mirrors exactly
            # what happens at real prediction time.
            scaled_data = {
                column: self.feature_scalers[column].transform(
                    features[column].values.reshape(-1, 1)
                )
                for column in features.columns
            }

            # Build sequences over the whole series
            X, y = [], []
            for i in range(sequence_length, n_samples):
                sequence = [scaled_data[column][i - sequence_length:i] for column in features.columns]
                X.append(np.column_stack(sequence))
                # Target must be in the same scaled space the model's output
                # is later inverse-transformed from - using the raw price
                # here would train the network against one scale while
                # predict()/evaluate() inverse-transform as if it were scaled.
                y.append(scaled_data[target_column][i, 0])
            X, y = np.array(X), np.array(y)

            # Map raw-row split boundaries onto the sequence arrays (sequence i
            # predicts raw row i + sequence_length)
            val_seq_idx = max(0, train_idx - sequence_length)
            test_seq_idx = max(0, test_idx - sequence_length)

            X_train, y_train = X[:val_seq_idx], y[:val_seq_idx]
            X_val, y_val = X[val_seq_idx:test_seq_idx], y[val_seq_idx:test_seq_idx]
            X_test, y_test = X[test_seq_idx:], y[test_seq_idx:]

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise

    def prepare_prediction_data(self,
                              df: pd.DataFrame,
                              sequence_length: int = Config.MODEL_SEQUENCE_LENGTH) -> np.ndarray:
        """
        Prepare the most recent sequence for live prediction, using
        already-fitted scalers (from training or loaded from disk).

        Args:
            df: Input DataFrame
            sequence_length: Length of input sequences

        Returns:
            np.ndarray: Prepared input data, shape (1, sequence_length, n_features)
        """
        try:
            if not self.feature_scalers:
                raise ValidationError("Scalers are not fitted; train or load a model first")

            features = self._build_features(df)
            if len(features) < sequence_length:
                raise ValidationError(f"Not enough data points. Need at least {sequence_length}")

            latest_data = features.tail(sequence_length)

            missing = [c for c in self.feature_columns if c not in latest_data.columns]
            if missing:
                raise ValidationError(f"Input data is missing columns the model was trained on: {missing}")

            scaled_sequence = []
            for column in self.feature_columns:
                scaler = self.feature_scalers[column]
                values = latest_data[column].values.reshape(-1, 1)
                scaled_sequence.append(scaler.transform(values))

            X = np.column_stack(scaled_sequence)
            return np.array([X])

        except Exception as e:
            logger.error(f"Error preparing prediction data: {str(e)}")
            raise

    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Convert scaled predictions back to original scale

        Args:
            predictions: Scaled predictions

        Returns:
            np.ndarray: Original scale predictions
        """
        try:
            if 'Close' not in self.feature_scalers:
                raise ValidationError("Price scaler not fitted")

            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)

            return self.feature_scalers['Close'].inverse_transform(predictions)

        except Exception as e:
            logger.error(f"Error inverse transforming predictions: {str(e)}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        flat = (gain == 0) & (loss == 0)
        return (100 - (100 / (1 + rs))).fillna(100).where(~flat, 50)

    def save_scalers(self, path: str, extra: Optional[Dict[str, Any]] = None):
        """Save fitted scalers, plus any extra metadata (e.g. sequence_length) the caller wants persisted alongside them"""
        try:
            import joblib
            scaler_dict = {
                'feature_scalers': self.feature_scalers,
                'feature_columns': self.feature_columns,
                **(extra or {})
            }
            joblib.dump(scaler_dict, path)

        except Exception as e:
            logger.error(f"Error saving scalers: {str(e)}")
            raise

    def load_scalers(self, path: str) -> Dict[str, Any]:
        """Load saved scalers. Returns the full saved dict so callers can read back any extra metadata."""
        try:
            import joblib
            scaler_dict = joblib.load(path)
            self.feature_scalers = scaler_dict['feature_scalers']
            self.feature_columns = scaler_dict.get('feature_columns', list(self.feature_scalers.keys()))
            return scaler_dict

        except Exception as e:
            logger.error(f"Error loading scalers: {str(e)}")
            raise
