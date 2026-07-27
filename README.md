# Stock Market Analysis & Prediction Application

A comprehensive stock market analysis and prediction tool with machine learning capabilities and a modern GUI interface.

## Features

- Real-time stock data visualization
- Technical analysis with multiple indicators
- Machine learning-based price predictions
- Portfolio management
- Advanced technical analysis tools

## Requirements

- Python 3.8 or higher
- Required packages listed in `requirements.txt`

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

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Activate the virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Run the application:
```bash
python run.py
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
Stock&Finance/
├── data/               # Data storage
├── models/            # Trained ML models
├── src/               # Source code
│   ├── gui/           # GUI components
│   │   ├── widgets/   # Custom widgets
│   │   └── styles/    # QSS stylesheets
│   └── utils/         # Utility functions
├── requirements.txt   # Package dependencies
├── README.md         # This file
└── run.py            # Application launcher
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.