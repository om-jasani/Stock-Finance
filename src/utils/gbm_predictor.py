"""Gradient-boosted tree price predictor (LightGBM).

Tree-based models on engineered tabular/lag features are a well-documented
strong, fast baseline for financial time series - often matching or beating
a deep sequence model while training in a fraction of the time, with no
GPU required. This is offered as a second, selectable model type alongside
the LSTM in StockPredictor rather than a replacement for it.

Unlike the LSTM path, no windowed sequences or feature scaling are needed:
LightGBM is scale-invariant (it splits on raw feature values), so the only
preprocessing is lag-feature construction and a chronological train/test
split - both point-in-time by construction, so there is no separate
"fit on train only" scaling step required to avoid leakage here.
"""
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import Config
from .exceptions import ModelError, ModelPredictionError, ModelTrainingError, ValidationError
from .logger import logger

_BASE_FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI']
_LAG_COLUMNS = ['Close', 'Volume', 'RSI']
_LAGS = [1, 2, 3, 5, 10]


class GBMPredictor:
    """LightGBM-based next-day return predictor"""

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[lgb.LGBMRegressor] = None
        self.model_path = model_path or os.path.join(Config.MODELS_DIR, 'gbm_predictor.pkl')
        self.feature_columns: List[str] = []
        self.used_gpu = False

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build tabular features with lagged columns for temporal context"""
        missing = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c not in df.columns]
        if missing:
            raise ValidationError(f"Missing required columns: {missing}")

        features = df.copy()
        if 'SMA_20' not in features.columns:
            features['SMA_20'] = features['Close'].rolling(window=20).mean()
        if 'SMA_50' not in features.columns:
            features['SMA_50'] = features['Close'].rolling(window=50).mean()
        if 'RSI' not in features.columns:
            delta = features['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            features['RSI'] = (100 - (100 / (1 + rs))).fillna(100)

        for column in _LAG_COLUMNS:
            for lag in _LAGS:
                features[f'{column}_lag_{lag}'] = features[column].shift(lag)

        feature_columns = _BASE_FEATURE_COLUMNS + [
            f'{c}_lag_{l}' for c in _LAG_COLUMNS for l in _LAGS
        ]
        return features[feature_columns].copy(), features['Close']

    def train(self,
             df: pd.DataFrame,
             val_size: float = 0.15,
             test_size: float = 0.15,
             params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train a next-day-return regressor with early stopping"""
        try:
            features, close = self._build_features(df)
            # Predict next-day return rather than raw price: stationary,
            # avoids the trivial "predict == today" shortcut, and is easy
            # to turn back into a price via last_close * (1 + return).
            target = close.pct_change().shift(-1)

            data = pd.concat([features, target.rename('target')], axis=1).dropna()
            self.feature_columns = list(features.columns)

            n = len(data)
            min_required = 50
            if n < min_required:
                raise ValidationError(f"Not enough data to train: got {n} usable rows, need at least {min_required}")

            test_idx = int(n * (1 - test_size))
            train_idx = int(test_idx * (1 - val_size))

            X, y = data[self.feature_columns], data['target']
            X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
            X_val, y_val = X.iloc[train_idx:test_idx], y.iloc[train_idx:test_idx]
            X_test, y_test = X.iloc[test_idx:], y.iloc[test_idx:]

            model_params = {
                'n_estimators': 500,
                'learning_rate': 0.03,
                'num_leaves': 31,
                'max_depth': -1,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                **(params or {})
            }

            self.model, self.used_gpu = self._fit_with_gpu_fallback(model_params, X_train, y_train, X_val, y_val)

            test_pred = self.model.predict(X_test)
            test_mse = float(np.mean((y_test.values - test_pred) ** 2))
            direction_accuracy = float(
                np.mean(np.sign(test_pred) == np.sign(y_test.values)) * 100
            ) if len(y_test) else 0.0

            return {
                'test_loss': test_mse,
                'direction_accuracy': direction_accuracy,
                'best_iteration': getattr(self.model, 'best_iteration_', model_params['n_estimators']),
                'used_gpu': self.used_gpu
            }

        except Exception as e:
            logger.error(f"Error training GBM model: {str(e)}")
            raise ModelTrainingError(f"Failed to train GBM model: {str(e)}")

    def _fit_with_gpu_fallback(self, model_params, X_train, y_train, X_val, y_val):
        """Try LightGBM's GPU device, falling back to CPU if unavailable"""
        for device in ('gpu', 'cpu'):
            try:
                model = lgb.LGBMRegressor(device=device, verbosity=-1, **model_params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )
                if device == 'gpu':
                    logger.info("LightGBM training used GPU")
                return model, device == 'gpu'
            except lgb.basic.LightGBMError as e:
                if device == 'gpu':
                    logger.info(f"LightGBM GPU unavailable ({e}); falling back to CPU")
                    continue
                raise

    def save(self, model_path: Optional[str] = None) -> None:
        """Persist the trained booster and its feature schema"""
        if self.model is None:
            raise ModelError("No model to save; train a model first")

        model_path = model_path or self.model_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({'model': self.model, 'feature_columns': self.feature_columns}, model_path)
        self.model_path = model_path

    def predict_future(self, df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """Predict future closes by iteratively applying the predicted daily return"""
        try:
            if self.model is None:
                if not self.load_model():
                    raise ModelError("No trained GBM model available")

            working_df = df.copy()
            predictions = []

            for _ in range(days):
                features, close = self._build_features(working_df)
                row = features[self.feature_columns].iloc[[-1]]
                predicted_return = float(self.model.predict(row)[0])

                last_close = close.iloc[-1]
                next_close = last_close * (1 + predicted_return)
                predictions.append(next_close)

                next_row = working_df.iloc[[-1]].copy()
                next_index = working_df.index[-1] + (working_df.index[-1] - working_df.index[-2])
                next_row.index = [next_index]
                next_row['Close'] = next_close
                working_df = pd.concat([working_df, next_row])

            last_date = pd.to_datetime(df.index[-1])
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
            return pd.DataFrame(index=future_dates, data={'Predicted_Close': predictions})

        except Exception as e:
            logger.error(f"Error predicting future prices with GBM: {str(e)}")
            raise ModelPredictionError(f"Failed to predict future prices: {str(e)}")

    def load_model(self) -> bool:
        """Load a saved GBM model"""
        try:
            if not os.path.exists(self.model_path):
                return False
            saved = joblib.load(self.model_path)
            self.model = saved['model']
            self.feature_columns = saved['feature_columns']
            return True
        except Exception as e:
            logger.error(f"Error loading GBM model: {str(e)}")
            raise ModelError(f"Failed to load GBM model: {str(e)}")
