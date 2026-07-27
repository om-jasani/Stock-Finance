#!/usr/bin/env python3
"""Headless model training - no GUI/display required.

Runs the exact same StockPredictor/GBMPredictor/ModelManager code the
desktop app uses, so a trained model shows up in the app immediately and
vice versa. Intended for unattended use on a training server, cron job,
or CI runner (e.g. inside the Dockerfile in this repo).

Examples:
    python scripts/train_model.py --symbol AAPL --model-type lstm --years 5
    python scripts/train_model.py --symbol AAPL --model-type gbm --tune --trials 30
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_fetcher import DataFetcher
from src.utils.model_trainer import StockPredictor
from src.utils.gbm_predictor import GBMPredictor
from src.utils.model_manager import ModelManager
from src.utils.hyperparameter_search import tune_lstm, tune_gbm
from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--symbol', required=True, help='Stock symbol, e.g. AAPL')
    parser.add_argument('--model-type', choices=['lstm', 'gbm'], default='lstm')
    parser.add_argument('--years', type=int, default=2, help='Years of history to train on')
    parser.add_argument('--epochs', type=int, default=None, help='LSTM only; defaults to Config.MODEL_EPOCHS')
    parser.add_argument('--tune', action='store_true', help='Run Optuna hyperparameter search first')
    parser.add_argument('--trials', type=int, default=20, help='Optuna trial count when --tune is set')
    args = parser.parse_args()

    fetcher = DataFetcher()
    df = fetcher.get_stock_data(args.symbol, period=f"{args.years}y", interval='1d')
    if df is None or df.empty:
        logger.error(f"No data available for {args.symbol}")
        sys.exit(1)

    if args.model_type == 'lstm':
        model_kwargs = {}
        if args.tune:
            search = tune_lstm(df, n_trials=args.trials)
            best = search['best_params']
            model_kwargs = dict(
                sequence_length=best['sequence_length'],
                hidden_sizes=(best['hidden1'], best['hidden2'], best['hidden3']),
                dropout=best['dropout'],
                learning_rate=best['learning_rate'],
            )
            print(f"Tuned hyperparameters: {best}")

        model = StockPredictor(**model_kwargs)
        train_kwargs = {}
        if args.epochs:
            train_kwargs['epochs'] = args.epochs
        results = model.train(df, **train_kwargs)
    else:
        params = None
        if args.tune:
            search = tune_gbm(df, n_trials=args.trials)
            params = {k: v for k, v in search['best_params'].items()}
            print(f"Tuned hyperparameters: {params}")

        model = GBMPredictor()
        results = model.train(df, params=params)

    manager = ModelManager()
    model_id = manager.save_model(
        symbol=args.symbol, model=model, metrics=results, model_type=args.model_type
    )

    print(f"Trained and saved model: {model_id}")
    print(f"Metrics: { {k: v for k, v in results.items() if k != 'history'} }")


if __name__ == '__main__':
    main()
