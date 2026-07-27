"""Widget management and registration"""
from PyQt6.QtWidgets import QWidget
from typing import Dict, Type, Optional
from .widgets import (
    StockChartWidget,
    PredictionWidget,
    PortfolioWidget,
    AnalysisWidget
)

class WidgetManager:
    """Manage GUI widgets and their states"""
    
    def __init__(self):
        self._widgets: Dict[str, QWidget] = {}
        self._current_symbol: Optional[str] = None
        self._register_widgets()
        
    def _register_widgets(self):
        """Register all available widgets"""
        self._widgets = {
            'chart': StockChartWidget(),
            'prediction': PredictionWidget(),
            'portfolio': PortfolioWidget(),
            'analysis': AnalysisWidget()
        }
        
    def get_widget(self, name: str) -> Optional[QWidget]:
        """Get widget by name"""
        return self._widgets.get(name)
        
    def update_symbol(self, symbol: str):
        """Update current symbol across all widgets"""
        self._current_symbol = symbol
        for widget in self._widgets.values():
            if hasattr(widget, 'update_symbol'):
                widget.update_symbol(symbol)
                
    def get_current_symbol(self) -> Optional[str]:
        """Get current symbol"""
        return self._current_symbol