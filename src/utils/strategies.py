"""Predefined trading strategies"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .logger import logger

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize strategy with parameters"""
        self.params = params or {}
        
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal"""
        raise NotImplementedError("Subclasses must implement generate_signal")

class MovingAverageCrossover(TradingStrategy):
    """Moving average crossover strategy"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'short_window': 20,
            'long_window': 50
        }
        super().__init__({**default_params, **(params or {})})
        
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            if len(data) < self.params['long_window']:
                return {'action': 'HOLD'}
                
            # Calculate moving averages
            short_ma = data['Close'].rolling(window=self.params['short_window']).mean()
            long_ma = data['Close'].rolling(window=self.params['long_window']).mean()
            
            # Generate signals
            if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
                return {
                    'action': 'BUY',
                    'reason': 'Golden Cross',
                    'short_ma': short_ma.iloc[-1],
                    'long_ma': long_ma.iloc[-1]
                }
            elif short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]:
                return {
                    'action': 'SELL',
                    'reason': 'Death Cross',
                    'short_ma': short_ma.iloc[-1],
                    'long_ma': long_ma.iloc[-1]
                }
                
            return {'action': 'HOLD'}
            
        except Exception as e:
            logger.error(f"Error in MA Crossover strategy: {str(e)}")
            return {'action': 'HOLD'}

class RSIStrategy(TradingStrategy):
    """RSI-based trading strategy"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'rsi_period': 14,
            'overbought': 70,
            'oversold': 30
        }
        super().__init__({**default_params, **(params or {})})
        
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            if len(data) < self.params['rsi_period']:
                return {'action': 'HOLD'}
                
            current_rsi = data['RSI'].iloc[-1]
            prev_rsi = data['RSI'].iloc[-2]
            
            if current_rsi < self.params['oversold'] and prev_rsi >= self.params['oversold']:
                return {
                    'action': 'BUY',
                    'reason': 'RSI Oversold',
                    'rsi': current_rsi
                }
            elif current_rsi > self.params['overbought'] and prev_rsi <= self.params['overbought']:
                return {
                    'action': 'SELL',
                    'reason': 'RSI Overbought',
                    'rsi': current_rsi
                }
                
            return {'action': 'HOLD'}
            
        except Exception as e:
            logger.error(f"Error in RSI strategy: {str(e)}")
            return {'action': 'HOLD'}

class MACDStrategy(TradingStrategy):
    """MACD-based trading strategy"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'signal_threshold': 0
        }
        super().__init__({**default_params, **(params or {})})
        
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            if len(data) < 26:  # Minimum data needed for MACD
                return {'action': 'HOLD'}
                
            macd = data['MACD'].iloc[-1]
            signal = data['Signal_Line'].iloc[-1]
            prev_macd = data['MACD'].iloc[-2]
            prev_signal = data['Signal_Line'].iloc[-2]
            
            if macd > signal and prev_macd <= prev_signal:
                return {
                    'action': 'BUY',
                    'reason': 'MACD Bullish Crossover',
                    'macd': macd,
                    'signal': signal
                }
            elif macd < signal and prev_macd >= prev_signal:
                return {
                    'action': 'SELL',
                    'reason': 'MACD Bearish Crossover',
                    'macd': macd,
                    'signal': signal
                }
                
            return {'action': 'HOLD'}
            
        except Exception as e:
            logger.error(f"Error in MACD strategy: {str(e)}")
            return {'action': 'HOLD'}

class BollingerBandsStrategy(TradingStrategy):
    """Bollinger Bands trading strategy"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'bb_period': 20,
            'bb_std': 2
        }
        super().__init__({**default_params, **(params or {})})
        
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            if len(data) < self.params['bb_period']:
                return {'action': 'HOLD'}
                
            close = data['Close'].iloc[-1]
            bb_lower = data['BB_lower'].iloc[-1]
            bb_upper = data['BB_upper'].iloc[-1]
            
            if close <= bb_lower:
                return {
                    'action': 'BUY',
                    'reason': 'Price at Lower BB',
                    'bb_lower': bb_lower,
                    'price': close
                }
            elif close >= bb_upper:
                return {
                    'action': 'SELL',
                    'reason': 'Price at Upper BB',
                    'bb_upper': bb_upper,
                    'price': close
                }
                
            return {'action': 'HOLD'}
            
        except Exception as e:
            logger.error(f"Error in Bollinger Bands strategy: {str(e)}")
            return {'action': 'HOLD'}

class TrendFollowingStrategy(TradingStrategy):
    """Trend following strategy combining multiple indicators"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'rsi_period': 14,
            'rsi_threshold': 50,
            'trend_period': 20,
            'volume_ma_period': 20
        }
        super().__init__({**default_params, **(params or {})})
        
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            if len(data) < max(self.params['trend_period'], self.params['rsi_period']):
                return {'action': 'HOLD'}
                
            # Calculate trend indicators
            current_price = data['Close'].iloc[-1]
            sma = data['SMA_20'].iloc[-1]
            rsi = data['RSI'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            volume_ma = data['Volume_SMA'].iloc[-1]
            
            # Count bullish signals
            bullish_signals = 0
            bearish_signals = 0
            
            # Price above SMA is bullish
            if current_price > sma:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            # RSI above threshold is bullish
            if rsi > self.params['rsi_threshold']:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            # Higher volume is confirming
            if volume > volume_ma:
                if current_price > data['Close'].iloc[-2]:  # Price increasing
                    bullish_signals += 1
                else:
                    bearish_signals += 1
                    
            # Generate signal based on majority
            if bullish_signals > bearish_signals:
                return {
                    'action': 'BUY',
                    'reason': 'Trend Following',
                    'bullish_signals': bullish_signals,
                    'total_signals': bullish_signals + bearish_signals
                }
            elif bearish_signals > bullish_signals:
                return {
                    'action': 'SELL',
                    'reason': 'Trend Following',
                    'bearish_signals': bearish_signals,
                    'total_signals': bullish_signals + bearish_signals
                }
                
            return {'action': 'HOLD'}
            
        except Exception as e:
            logger.error(f"Error in Trend Following strategy: {str(e)}")
            return {'action': 'HOLD'}

class MeanReversionStrategy(TradingStrategy):
    """Mean reversion strategy"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'lookback_period': 20,
            'std_dev_threshold': 2
        }
        super().__init__({**default_params, **(params or {})})
        
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            if len(data) < self.params['lookback_period']:
                return {'action': 'HOLD'}
                
            # Calculate rolling mean and standard deviation
            rolling_mean = data['Close'].rolling(window=self.params['lookback_period']).mean()
            rolling_std = data['Close'].rolling(window=self.params['lookback_period']).std()
            
            current_price = data['Close'].iloc[-1]
            mean = rolling_mean.iloc[-1]
            upper_band = mean + (rolling_std.iloc[-1] * self.params['std_dev_threshold'])
            lower_band = mean - (rolling_std.iloc[-1] * self.params['std_dev_threshold'])
            
            # Generate signals
            if current_price <= lower_band:
                return {
                    'action': 'BUY',
                    'reason': 'Price Below Mean',
                    'mean': mean,
                    'current_price': current_price,
                    'distance': (mean - current_price) / rolling_std.iloc[-1]
                }
            elif current_price >= upper_band:
                return {
                    'action': 'SELL',
                    'reason': 'Price Above Mean',
                    'mean': mean,
                    'current_price': current_price,
                    'distance': (current_price - mean) / rolling_std.iloc[-1]
                }
                
            return {'action': 'HOLD'}
            
        except Exception as e:
            logger.error(f"Error in Mean Reversion strategy: {str(e)}")
            return {'action': 'HOLD'}