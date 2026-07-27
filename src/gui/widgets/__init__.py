"""Initialize widgets package"""
from .stock_chart import StockChartWidget
from .prediction_widget import PredictionWidget
from .portfolio_widget import PortfolioWidget
from .analysis_widget import AnalysisWidget

__all__ = [
    'StockChartWidget',
    'PredictionWidget',
    'PortfolioWidget',
    'AnalysisWidget'
]