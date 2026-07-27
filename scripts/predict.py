#!/usr/bin/env python3
"""Headless prediction - no GUI/display required.

Loads the latest saved model for a symbol and prints/exports a future
price forecast. Uses the same ModelManager/StockPredictor/GBMPredictor
code as the desktop app.

Example:
    python scripts/predict.py --symbol AAPL --model-type lstm --days 14 --csv out.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_fetcher import DataFetcher
from src.utils.model_trainer import StockPredictor
from src.utils.model_manager import ModelManager
from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--symbol', required=True, help='Stock symbol, e.g. AAPL')
    parser.add_argument('--model-type', choices=['lstm', 'gbm'], default='lstm')
    parser.add_argument('--days', type=int, default=30, help='Days to predict forward')
    parser.add_argument('--csv', default=None, help='Optional path to write the forecast as CSV')
    args = parser.parse_args()

    manager = ModelManager()
    model = manager.get_latest_model(args.symbol, args.model_type)
    if model is None:
        logger.error(f"No trained {args.model_type} model found for {args.symbol}. Train one first with train_model.py")
        sys.exit(1)

    fetcher = DataFetcher()
    df = fetcher.get_stock_data(args.symbol, period='60d', interval='1d')
    if df is None or df.empty:
        logger.error(f"Failed to fetch recent data for {args.symbol}")
        sys.exit(1)

    if isinstance(model, StockPredictor):
        predictions = model.predict_future(df, days=args.days, confidence_interval=True)
    else:
        predictions = model.predict_future(df, days=args.days)

    print(predictions.to_string())

    if args.csv:
        predictions.to_csv(args.csv)
        print(f"Wrote forecast to {args.csv}")


if __name__ == '__main__':
    main()
