"""Optuna-driven hyperparameter search for both model types.

Uses walk-forward (rolling-origin) cross-validation rather than a single
static split: each trial is scored on the average validation loss across
several chronological folds, so a hyperparameter set that only happens to
fit one particular historical window well doesn't win the search - the
same principle applied to the train-only scaler fit in data_pipeline.py.
"""
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd

from .config import Config
from .gbm_predictor import GBMPredictor
from .logger import logger
from .model_trainer import StockPredictor

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _walk_forward_folds(n_samples: int, n_folds: int = 3, min_train_fraction: float = 0.5
                        ) -> List[Tuple[float, float]]:
    """Yield (train_end_fraction, val_end_fraction) pairs of increasing, non-overlapping windows"""
    fold_span = (1.0 - min_train_fraction) / n_folds
    return [
        (min_train_fraction + i * fold_span, min_train_fraction + (i + 1) * fold_span)
        for i in range(n_folds)
    ]


def tune_lstm(df: pd.DataFrame, n_trials: int = 20, n_folds: int = 3,
              epochs_per_trial: int = 20) -> Dict[str, Any]:
    """Search LSTM hyperparameters, scored by mean validation loss across walk-forward folds"""
    folds = _walk_forward_folds(len(df), n_folds)

    def objective(trial: optuna.Trial) -> float:
        sequence_length = trial.suggest_int('sequence_length', 20, 90, step=10)
        hidden1 = trial.suggest_int('hidden1', 32, 160, step=16)
        hidden2 = trial.suggest_int('hidden2', 16, hidden1, step=8)
        hidden3 = trial.suggest_int('hidden3', 8, hidden2, step=8)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

        fold_losses = []
        for train_end, val_end in folds:
            train_df = df.iloc[:int(len(df) * train_end)]
            if len(train_df) < sequence_length + 20:
                continue

            predictor = StockPredictor(
                hidden_sizes=(hidden1, hidden2, hidden3),
                dropout=dropout,
                sequence_length=sequence_length,
                learning_rate=learning_rate,
            )
            try:
                results = predictor.train(train_df, epochs=epochs_per_trial, batch_size=batch_size, patience=5)
                fold_losses.append(results['test_loss'])
            except Exception as e:
                logger.warning(f"Trial fold failed ({e}); penalizing")
                fold_losses.append(float('inf'))

        return float(np.mean(fold_losses)) if fold_losses else float('inf')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best LSTM trial: {study.best_value:.4f} with params {study.best_params}")
    return {'best_params': study.best_params, 'best_value': study.best_value}


def tune_gbm(df: pd.DataFrame, n_trials: int = 30, n_folds: int = 3) -> Dict[str, Any]:
    """Search LightGBM hyperparameters, scored by mean validation loss across walk-forward folds"""
    folds = _walk_forward_folds(len(df), n_folds)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators': 500,
        }

        fold_losses = []
        for train_end, val_end in folds:
            train_df = df.iloc[:int(len(df) * train_end)]
            if len(train_df) < 60:
                continue

            predictor = GBMPredictor()
            try:
                results = predictor.train(train_df, params=params)
                fold_losses.append(results['test_loss'])
            except Exception as e:
                logger.warning(f"Trial fold failed ({e}); penalizing")
                fold_losses.append(float('inf'))

        return float(np.mean(fold_losses)) if fold_losses else float('inf')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best GBM trial: {study.best_value:.6f} with params {study.best_params}")
    return {'best_params': study.best_params, 'best_value': study.best_value}
