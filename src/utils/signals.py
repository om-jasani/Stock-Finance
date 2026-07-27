"""Trading signals and alerts management"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from .market_analysis import MarketAnalyzer
from .data_fetcher import DataFetcher
from .logger import logger

class SignalGenerator:
    """Class for generating and managing trading signals"""
    
    def __init__(self):
        """Initialize SignalGenerator"""
        self.data_fetcher = DataFetcher()
        self.market_analyzer = MarketAnalyzer()
        self.active_signals: Set[str] = set()
        
    def generate_signals(self,
                        symbol: str,
                        indicators: Optional[List[str]] = None,
                        period: str = '1mo') -> Dict[str, Any]:
        """
        Generate trading signals for a symbol

        Args:
            symbol: Stock symbol
            indicators: List of indicators to use (default: all)
            period: Data period to fetch/analyze

        Returns:
            Dict[str, Any]: Generated signals
        """
        try:
            # Fetch data
            df = self.data_fetcher.get_stock_data(symbol, period=period)
            if df is None or df.empty:
                raise ValueError(f"No data available for {symbol}")
                
            signals = []
            
            # Technical analysis
            trend_analysis = self.market_analyzer.analyze_trend(df)
            patterns = self.market_analyzer.analyze_patterns(df)
            support_resistance = self.market_analyzer.find_support_resistance(df)
            
            # Price signals
            current_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2]
            
            # Check price movement
            if current_price > prev_close:
                signals.append({
                    'type': 'PRICE',
                    'signal': 'Price increase',
                    'strength': 'Medium',
                    'direction': 'Bullish',
                    'details': f"Price up {((current_price/prev_close)-1)*100:.2f}%"
                })
            else:
                signals.append({
                    'type': 'PRICE',
                    'signal': 'Price decrease',
                    'strength': 'Medium',
                    'direction': 'Bearish',
                    'details': f"Price down {((prev_close/current_price)-1)*100:.2f}%"
                })

            # Support/Resistance signals
            for support in support_resistance['support']:
                if abs(current_price - support) / current_price < 0.02:
                    signals.append({
                        'type': 'SUPPORT',
                        'signal': 'Near support level',
                        'strength': 'High',
                        'direction': 'Bullish',
                        'details': f"Support at ${support:.2f}"
                    })

            for resistance in support_resistance['resistance']:
                if abs(current_price - resistance) / current_price < 0.02:
                    signals.append({
                        'type': 'RESISTANCE',
                        'signal': 'Near resistance level',
                        'strength': 'High',
                        'direction': 'Bearish',
                        'details': f"Resistance at ${resistance:.2f}"
                    })

            # RSI signals
            rsi = df['RSI'].iloc[-1]
            if rsi > 70:
                signals.append({
                    'type': 'RSI',
                    'signal': 'Overbought',
                    'strength': 'High',
                    'direction': 'Bearish',
                    'details': f"RSI at {rsi:.1f}"
                })
            elif rsi < 30:
                signals.append({
                    'type': 'RSI',
                    'signal': 'Oversold',
                    'strength': 'High',
                    'direction': 'Bullish',
                    'details': f"RSI at {rsi:.1f}"
                })

            # MACD signals
            macd = df['MACD'].iloc[-1]
            signal_line = df['Signal_Line'].iloc[-1]
            prev_macd = df['MACD'].iloc[-2]
            prev_signal = df['Signal_Line'].iloc[-2]

            if macd > signal_line and prev_macd <= prev_signal:
                signals.append({
                    'type': 'MACD',
                    'signal': 'Bullish crossover',
                    'strength': 'High',
                    'direction': 'Bullish',
                    'details': 'MACD crossed above signal line'
                })
            elif macd < signal_line and prev_macd >= prev_signal:
                signals.append({
                    'type': 'MACD',
                    'signal': 'Bearish crossover',
                    'strength': 'High',
                    'direction': 'Bearish',
                    'details': 'MACD crossed below signal line'
                })

            # Volume signals
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume_SMA'].iloc[-1]

            if current_volume > avg_volume * 2:
                signals.append({
                    'type': 'VOLUME',
                    'signal': 'High volume',
                    'strength': 'Medium',
                    'direction': 'Neutral',
                    'details': f"Volume {current_volume/avg_volume:.1f}x average"
                })

            # Pattern signals
            for pattern in patterns:
                if 'Bullish' in pattern['name']:
                    direction = 'Bullish'
                elif 'Bearish' in pattern['name'] or pattern['name'] == 'Hanging Man':
                    direction = 'Bearish'
                else:
                    direction = 'Neutral'
                signals.append({
                    'type': 'PATTERN',
                    'signal': pattern['name'],
                    'strength': pattern['confidence'],
                    'direction': direction,
                    'details': f"{pattern['type']} pattern detected"
                })

            # Bollinger Bands signals
            bb_upper = df['BB_upper'].iloc[-1]
            bb_lower = df['BB_lower'].iloc[-1]

            if current_price > bb_upper:
                signals.append({
                    'type': 'BOLLINGER',
                    'signal': 'Price above upper band',
                    'strength': 'Medium',
                    'direction': 'Bearish',
                    'details': 'Potential overbought condition'
                })
            elif current_price < bb_lower:
                signals.append({
                    'type': 'BOLLINGER',
                    'signal': 'Price below lower band',
                    'strength': 'Medium',
                    'direction': 'Bullish',
                    'details': 'Potential oversold condition'
                })

            # Trend signals
            if trend_analysis['trend'] in ['Strong Bullish', 'Strong Bearish']:
                signals.append({
                    'type': 'TREND',
                    'signal': trend_analysis['trend'],
                    'strength': 'High',
                    'direction': 'Bullish' if 'Bullish' in trend_analysis['trend'] else 'Bearish',
                    'details': f"Trend strength: {trend_analysis['trend_strength']:.1f}%"
                })

            # Filter signals based on requested indicators
            if indicators:
                signals = [s for s in signals if s['type'] in indicators]

            # Add metadata
            signal_summary = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price': current_price,
                'volume': current_volume,
                'volume_sma': avg_volume,
                'signals': signals,
                'total_signals': len(signals),
                'trend': trend_analysis['trend'],
                'support_levels': support_resistance['support'],
                'resistance_levels': support_resistance['resistance']
            }

            return signal_summary
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            raise
            
    def get_signal_strength(self, signals: List[Dict[str, Any]]) -> float:
        """Calculate overall signal strength"""
        try:
            if not signals:
                return 0.0
                
            # Assign weights to different signal strengths
            strength_weights = {
                'High': 1.0,
                'Medium': 0.6,
                'Low': 0.3
            }
            
            # Calculate weighted average of signal strengths
            total_weight = 0
            weighted_sum = 0
            
            for signal in signals:
                weight = strength_weights.get(signal['strength'], 0.5)
                total_weight += weight
                weighted_sum += weight * {
                    'Bullish': 1,
                    'Bearish': -1
                }.get(signal.get('direction', 'Neutral'), 0)
                
            if total_weight == 0:
                return 0.0
                
            return weighted_sum / total_weight
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {str(e)}")
            return 0.0
            
    def filter_signals(self,
                     signals: List[Dict[str, Any]],
                     min_strength: str = 'Medium',
                     types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Filter signals based on criteria"""
        try:
            strength_levels = {'Low': 0, 'Medium': 1, 'High': 2}
            min_strength_level = strength_levels[min_strength]
            
            filtered = [
                s for s in signals
                if strength_levels[s['strength']] >= min_strength_level
                and (not types or s['type'] in types)
            ]
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error filtering signals: {str(e)}")
            return []
            
    def consolidate_signals(self, symbol: str, timeframe: str = '1d') -> Dict[str, Any]:
        """Consolidate signals across different timeframes"""
        try:
            timeframes = {
                '1h': '1d',
                '4h': '5d',
                '1d': '1mo',
                '1w': '3mo'
            }
            
            consolidated_signals = {}
            overall_strength = 0
            
            for tf_label, period in timeframes.items():
                try:
                    signals = self.generate_signals(symbol, period=period)
                except Exception as e:
                    logger.warning(f"Skipping timeframe {tf_label} for {symbol}: {e}")
                    continue

                strength = self.get_signal_strength(signals['signals'])

                consolidated_signals[tf_label] = {
                    'signals': signals['signals'],
                    'strength': strength,
                    'trend': signals['trend']
                }

                # Weight longer timeframes more heavily
                timeframe_weights = {'1h': 0.1, '4h': 0.2, '1d': 0.3, '1w': 0.4}
                overall_strength += strength * timeframe_weights[tf_label]
                    
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'timeframes': consolidated_signals,
                'overall_strength': overall_strength,
                'overall_bias': 'Bullish' if overall_strength > 0 else 'Bearish'
            }
            
        except Exception as e:
            logger.error(f"Error consolidating signals: {str(e)}")
            raise