"""Market analysis utilities"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .data_fetcher import DataFetcher
from .exceptions import ValidationError
from .logger import logger

class MarketAnalyzer:
    """Class for performing market analysis"""
    
    def __init__(self):
        """Initialize MarketAnalyzer"""
        self.data_fetcher = DataFetcher()
        
    def analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price trend
        
        Args:
            df: Stock data DataFrame
            
        Returns:
            Dict[str, Any]: Trend analysis results
        """
        try:
            # Get last row of data
            current = df.iloc[-1]
            
            # Calculate trend indicators
            sma_20_trend = current['Close'] > current['SMA_20']
            sma_50_trend = current['Close'] > current['SMA_50']
            macd_trend = current['MACD'] > current['Signal_Line']
            rsi = current['RSI']
            
            # Determine trend strength
            trend_signals = [
                sma_20_trend,
                sma_50_trend,
                macd_trend,
                rsi > 50
            ]
            
            bullish_signals = sum(trend_signals)
            trend_strength = (bullish_signals / len(trend_signals)) * 100
            
            # Determine overall trend
            if trend_strength >= 75:
                trend = "Strong Bullish"
            elif trend_strength >= 50:
                trend = "Moderately Bullish"
            elif trend_strength >= 25:
                trend = "Moderately Bearish"
            else:
                trend = "Strong Bearish"
                
            # Calculate momentum
            momentum = df['Momentum'].iloc[-1]
            roc = df['ROC'].iloc[-1]
            
            # Volume analysis
            volume_trend = df['Volume'].iloc[-1] > df['Volume_SMA'].iloc[-1]
            
            return {
                'trend': trend,
                'trend_strength': trend_strength,
                'momentum': momentum,
                'roc': roc,
                'rsi': rsi,
                'volume_trend': 'Increasing' if volume_trend else 'Decreasing',
                'sma_20_trend': 'Above' if sma_20_trend else 'Below',
                'sma_50_trend': 'Above' if sma_50_trend else 'Below',
                'macd_signal': 'Bullish' if macd_trend else 'Bearish'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            raise
            
    def find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """
        Find support and resistance levels
        
        Args:
            df: Stock data DataFrame
            window: Window size for analysis
            
        Returns:
            Dict[str, List[float]]: Support and resistance levels
        """
        try:
            highs = df['High'].rolling(window=window, center=True).max()
            lows = df['Low'].rolling(window=window, center=True).min()
            
            # Find local maxima and minima
            resistance_levels = []
            support_levels = []
            
            for i in range(window, len(df) - window):
                if highs.iloc[i] == df['High'].iloc[i]:
                    resistance_levels.append(df['High'].iloc[i])
                if lows.iloc[i] == df['Low'].iloc[i]:
                    support_levels.append(df['Low'].iloc[i])
                    
            # Group close levels together
            def group_levels(levels, threshold=0.02):
                if not levels:
                    return []
                    
                levels = sorted(levels)
                grouped = []
                current_group = [levels[0]]
                
                for level in levels[1:]:
                    if (level - current_group[-1]) / current_group[-1] <= threshold:
                        current_group.append(level)
                    else:
                        grouped.append(sum(current_group) / len(current_group))
                        current_group = [level]
                        
                grouped.append(sum(current_group) / len(current_group))
                return grouped
                
            # Get most significant levels
            resistance_levels = group_levels(resistance_levels)[-3:]  # Top 3 resistance
            support_levels = group_levels(support_levels)[:3]  # Top 3 support
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {str(e)}")
            raise
            
    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """
        Calculate volatility metrics
        
        Args:
            df: Stock data DataFrame
            window: Window size for calculations
            
        Returns:
            Dict[str, float]: Volatility metrics
        """
        try:
            # Calculate daily returns
            returns = df['Close'].pct_change()
            
            # Historical volatility (standard deviation of returns)
            historical_volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Rolling volatility
            rolling_volatility = returns.rolling(window=window).std() * np.sqrt(252)
            current_volatility = rolling_volatility.iloc[-1]
            
            # Average True Range
            atr = df['ATR'].iloc[-1]
            
            # Bollinger Band Width
            bb_width = (df['BB_upper'].iloc[-1] - df['BB_lower'].iloc[-1]) / df['BB_middle'].iloc[-1]
            
            return {
                'historical_volatility': historical_volatility,
                'current_volatility': current_volatility,
                'atr': atr,
                'bollinger_width': bb_width
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            raise
            
    def analyze_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect technical patterns
        
        Args:
            df: Stock data DataFrame
            
        Returns:
            List[Dict[str, Any]]: Detected patterns
        """
        try:
            patterns = []
            
            # Get recent data points
            recent_data = df.tail(5)
            closes = recent_data['Close'].values
            highs = recent_data['High'].values
            lows = recent_data['Low'].values
            
            # Doji pattern
            body_size = abs(closes[-1] - recent_data['Open'].iloc[-1])
            wick_size = highs[-1] - lows[-1]
            if body_size < (wick_size * 0.1):
                patterns.append({
                    'name': 'Doji',
                    'type': 'Reversal',
                    'confidence': 'High' if body_size < (wick_size * 0.05) else 'Medium'
                })
            
            # Hammer/Hanging Man
            body_range = abs(closes[-1] - recent_data['Open'].iloc[-1])
            upper_wick = highs[-1] - max(closes[-1], recent_data['Open'].iloc[-1])
            lower_wick = min(closes[-1], recent_data['Open'].iloc[-1]) - lows[-1]
            
            if (body_range > 0 and
                ((lower_wick > (body_range * 2) and upper_wick < (body_range * 0.1)) or
                 (upper_wick > (body_range * 2) and lower_wick < (body_range * 0.1)))):
                patterns.append({
                    'name': 'Hammer' if df['Close'].pct_change().iloc[-1] > 0 else 'Hanging Man',
                    'type': 'Reversal',
                    'confidence': 'Medium'
                })
            
            # Engulfing patterns
            if len(closes) >= 2:
                curr_body_range = abs(closes[-1] - recent_data['Open'].iloc[-1])
                prev_body_range = abs(closes[-2] - recent_data['Open'].iloc[-2])
                
                if curr_body_range > prev_body_range:
                    if closes[-1] > recent_data['Open'].iloc[-1] and closes[-2] < recent_data['Open'].iloc[-2]:
                        patterns.append({
                            'name': 'Bullish Engulfing',
                            'type': 'Reversal',
                            'confidence': 'High'
                        })
                    elif closes[-1] < recent_data['Open'].iloc[-1] and closes[-2] > recent_data['Open'].iloc[-2]:
                        patterns.append({
                            'name': 'Bearish Engulfing',
                            'type': 'Reversal',
                            'confidence': 'High'
                        })
            
            # Price Channels
            window = 20
            upper_channel = df['High'].rolling(window=window).max()
            lower_channel = df['Low'].rolling(window=window).min()
            
            if abs(closes[-1] - upper_channel.iloc[-1]) / closes[-1] < 0.01:
                patterns.append({
                    'name': 'Channel Resistance Test',
                    'type': 'Resistance',
                    'confidence': 'Medium'
                })
            elif abs(closes[-1] - lower_channel.iloc[-1]) / closes[-1] < 0.01:
                patterns.append({
                    'name': 'Channel Support Test',
                    'type': 'Support',
                    'confidence': 'Medium'
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            raise
            
    def get_trade_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals
        
        Args:
            df: Stock data DataFrame
            
        Returns:
            Dict[str, Any]: Trading signals and recommendations
        """
        try:
            current = df.iloc[-1]
            
            # Collect various signals
            signals = []
            confidence = 0
            
            # Moving Average signals
            if current['Close'] > current['SMA_20'] > current['SMA_50']:
                signals.append('Moving averages aligned bullishly')
                confidence += 20
            elif current['Close'] < current['SMA_20'] < current['SMA_50']:
                signals.append('Moving averages aligned bearishly')
                confidence -= 20
                
            # RSI signals
            if current['RSI'] > 70:
                signals.append('Overbought on RSI')
                confidence -= 15
            elif current['RSI'] < 30:
                signals.append('Oversold on RSI')
                confidence += 15
                
            # MACD signals
            if current['MACD'] > current['Signal_Line']:
                signals.append('MACD bullish crossover')
                confidence += 10
            else:
                signals.append('MACD bearish crossover')
                confidence -= 10
                
            # Volume confirmation
            if current['Volume'] > current['Volume_SMA']:
                signals.append('Above average volume')
                confidence += 5
                
            # Bollinger Band signals
            if current['Close'] > current['BB_upper']:
                signals.append('Price above upper Bollinger Band')
                confidence -= 10
            elif current['Close'] < current['BB_lower']:
                signals.append('Price below lower Bollinger Band')
                confidence += 10
                
            # Determine recommendation
            if confidence >= 30:
                recommendation = 'Strong Buy'
            elif confidence >= 10:
                recommendation = 'Buy'
            elif confidence <= -30:
                recommendation = 'Strong Sell'
            elif confidence <= -10:
                recommendation = 'Sell'
            else:
                recommendation = 'Hold'
                
            return {
                'recommendation': recommendation,
                'confidence': abs(confidence),
                'signals': signals,
                'price_target': self._calculate_price_target(df),
                'stop_loss': self._calculate_stop_loss(df)
            }
            
        except Exception as e:
            logger.error(f"Error generating trade signals: {str(e)}")
            raise
            
    def _calculate_price_target(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate price targets"""
        try:
            current_price = df['Close'].iloc[-1]
            atr = df['ATR'].iloc[-1]
            
            return {
                'conservative': current_price + (atr * 2),
                'moderate': current_price + (atr * 3),
                'aggressive': current_price + (atr * 4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating price targets: {str(e)}")
            raise
            
    def _calculate_stop_loss(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate stop loss levels"""
        try:
            current_price = df['Close'].iloc[-1]
            atr = df['ATR'].iloc[-1]
            
            return {
                'tight': current_price - (atr * 1.5),
                'moderate': current_price - (atr * 2),
                'wide': current_price - (atr * 3)
            }
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            raise