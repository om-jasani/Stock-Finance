# Stock Market Analysis & Prediction Application

A comprehensive stock market analysis and prediction tool with machine learning capabilities and a modern GUI interface.

## Features

- Real-time stock data visualization
- Technical analysis with multiple indicators
- Machine learning-based price predictions
- Portfolio management
- Advanced technical analysis tools

## Requirements

- Python 3.10+ (matches the PyTorch/PyQt6 stack this project is tested against)
- An NVIDIA/CUDA, Apple Silicon (MPS), or CPU-only machine - GPU is detected and used automatically when present

## Installation

1. Clone this repository:
```bash
git clone https://github.com/om-jasani/Stock-Finance.git
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install PyTorch with the CUDA build matching your GPU (see
   [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)),
   e.g. for a recent NVIDIA GPU on Windows/Linux:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

4. Install the rest of the dependencies:
```bash
pip install -r requirements.txt        # desktop app (GUI + core)
# or, for a headless/server install with no GUI:
pip install -r requirements-core.txt
```

5. Copy `.env.example` to `.env` and fill in any API keys you have (Alpha
   Vantage/Finnhub are optional fallbacks used only if yfinance fails).

## Usage

### Desktop app

1. Activate the virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Run the application:
```bash
python run.py
```

### Headless training/prediction (no GUI)

Useful for a GPU server, a cron job, or CI - uses the exact same model
code as the desktop app, so models trained this way show up in the app
and vice versa:
```bash
python scripts/train_model.py --symbol AAPL --model-type lstm --years 5
python scripts/train_model.py --symbol AAPL --model-type gbm --tune --trials 30
python scripts/predict.py --symbol AAPL --days 14 --csv forecast.csv
```

### Docker (headless, CUDA)

```bash
docker build -t stock-finance .
docker run --gpus all stock-finance scripts/train_model.py --symbol AAPL
```

## Features

### Stock Charts
- Real-time price charts
- Multiple timeframes
- Volume analysis
- Various technical indicators

### Predictions
- Machine learning-based price predictions
- Customizable prediction timeframes
- Model training capabilities
- Accuracy metrics

### Portfolio Management
- Track multiple stocks
- Performance analysis
- Gain/loss tracking
- Portfolio statistics

### Technical Analysis
- Moving averages
- RSI indicator
- MACD
- Bollinger Bands
- Support/Resistance levels

## Project Structure

```
Stock-Finance/
├── data/                    # Data storage (parquet cache, portfolio.json)
├── models/                  # Trained ML models
├── scripts/                 # Headless CLI entry points (train_model.py, predict.py)
├── src/                     # Source code
│   ├── gui/                 # GUI components
│   │   ├── widgets/         # Custom widgets
│   │   ├── styles/          # QSS stylesheets
│   │   └── workers.py       # Shared QThread background-fetch worker
│   └── utils/                # Data fetching, ML pipeline, trading logic
│       ├── model_trainer.py         # PyTorch LSTM predictor
│       ├── gbm_predictor.py         # LightGBM predictor
│       ├── hyperparameter_search.py # Optuna walk-forward tuning
│       └── data_pipeline.py         # Leakage-safe feature/scaling pipeline
├── requirements.txt         # Full desktop app (-> requirements-gui.txt)
├── requirements-gui.txt     # GUI layer (-> requirements-core.txt)
├── requirements-core.txt    # Headless core dependencies
├── requirements-lock.txt    # Exact reproducible dependency versions
├── Dockerfile                # Headless CUDA training/inference image
├── README.md                 # This file
└── run.py                    # Desktop app launcher
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.