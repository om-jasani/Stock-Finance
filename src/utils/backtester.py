"""Backtesting utilities for trading strategies"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from .data_fetcher import DataFetcher
from .market_analysis import MarketAnalyzer
from .exceptions import ValidationError
from .logger import logger

class BacktestResult:
    """Class to hold backtest results"""
    
    def __init__(self, trades: pd.DataFrame, metrics: Dict[str, Any]):
        self.trades = trades
        self.metrics = metrics
        
    def __str__(self) -> str:
        return f"Backtest Results:\n" + \
               f"Total Return: {self.metrics['total_return']:.2f}%\n" + \
               f"Annual Return: {self.metrics['annual_return']:.2f}%\n" + \
               f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}\n" + \
               f"Max Drawdown: {self.metrics['max_drawdown']:.2f}%\n" + \
               f"Win Rate: {self.metrics['win_rate']:.2f}%"

class Backtester:
    """Class for backtesting trading strategies"""
    
    def __init__(self, initial_capital: float = 100000):
        self.data_fetcher = DataFetcher()
        self.market_analyzer = MarketAnalyzer()
        self.initial_capital = initial_capital
        
    def run_backtest(self,
                    symbol: str,
                    strategy: Callable[[pd.DataFrame], Dict[str, Any]],
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    position_size: float = 0.1,
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None) -> BacktestResult:
        """Run backtest for a trading strategy"""
        try:
            # Fetch historical data
            df = self.data_fetcher.get_stock_data(symbol, period='max')
            
            if df is None or df.empty:
                raise ValidationError(f"No data available for {symbol}")
                
            # Filter date range if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            # Initialize variables
            capital = self.initial_capital
            position = 0
            trades = []
            current_trade = None
            
            # Run strategy
            for i in range(len(df)):
                current_data = df.iloc[:i+1]
                signal = strategy(current_data)
                
                price = current_data['Close'].iloc[-1]
                date = current_data.index[-1]
                
                # Process signals
                if signal.get('action') == 'BUY' and position == 0:
                    # Calculate position size
                    shares = (capital * position_size) // price
                    if shares > 0:
                        cost = shares * price
                        capital -= cost
                        position = shares
                        current_trade = {
                            'entry_date': date,
                            'entry_price': price,
                            'shares': shares,
                            'type': 'LONG'
                        }
                        
                elif signal.get('action') == 'SELL' and position > 0:
                    # Close position
                    revenue = position * price
                    capital += revenue
                    profit = revenue - (current_trade['shares'] * current_trade['entry_price'])
                    trades.append({
                        **current_trade,
                        'exit_date': date,
                        'exit_price': price,
                        'profit': profit,
                        'return': (profit / (current_trade['shares'] * current_trade['entry_price'])) * 100
                    })
                    position = 0
                    current_trade = None
                    
                # Check stop loss and take profit
                elif position > 0:
                    entry_price = current_trade['entry_price']
                    current_return = (price - entry_price) / entry_price
                    
                    if stop_loss and current_return <= -stop_loss:
                        # Stop loss hit
                        revenue = position * price
                        capital += revenue
                        profit = revenue - (current_trade['shares'] * current_trade['entry_price'])
                        trades.append({
                            **current_trade,
                            'exit_date': date,
                            'exit_price': price,
                            'profit': profit,
                            'return': (profit / (current_trade['shares'] * current_trade['entry_price'])) * 100,
                            'exit_reason': 'STOP_LOSS'
                        })
                        position = 0
                        current_trade = None
                        
                    elif take_profit and current_return >= take_profit:
                        # Take profit hit
                        revenue = position * price
                        capital += revenue
                        profit = revenue - (current_trade['shares'] * current_trade['entry_price'])
                        trades.append({
                            **current_trade,
                            'exit_date': date,
                            'exit_price': price,
                            'profit': profit,
                            'return': (profit / (current_trade['shares'] * current_trade['entry_price'])) * 100,
                            'exit_reason': 'TAKE_PROFIT'
                        })
                        position = 0
                        current_trade = None
                        
            # Close any remaining position
            if position > 0:
                price = df['Close'].iloc[-1]
                revenue = position * price
                capital += revenue
                profit = revenue - (current_trade['shares'] * current_trade['entry_price'])
                trades.append({
                    **current_trade,
                    'exit_date': df.index[-1],
                    'exit_price': price,
                    'profit': profit,
                    'return': (profit / (current_trade['shares'] * current_trade['entry_price'])) * 100,
                    'exit_reason': 'END_OF_PERIOD'
                })
                
            # Calculate metrics
            trades_df = pd.DataFrame(trades)
            metrics = self._calculate_metrics(trades_df, df, capital)
            
            return BacktestResult(trades_df, metrics)
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise
            
    def _calculate_metrics(self, trades_df: pd.DataFrame, price_data: pd.DataFrame, final_capital: float) -> Dict[str, Any]:
        """Calculate backtest metrics"""
        try:
            if trades_df.empty:
                return {
                    'total_return': 0,
                    'annual_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'number_of_trades': 0,
                    'average_return': 0,
                    'max_consecutive_losses': 0
                }
                
            # Basic metrics
            total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
            trading_days = (price_data.index[-1] - price_data.index[0]).days
            annual_return = (total_return / trading_days) * 365
            
            # Calculate returns series
            returns_series = pd.Series(index=price_data.index, data=0.0)
            for _, trade in trades_df.iterrows():
                returns_series[trade['exit_date']] = trade['return']
                
            # Sharpe ratio
            daily_returns = returns_series[returns_series != 0]
            if len(daily_returns) > 0:
                sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
            else:
                sharpe_ratio = 0
                
            # Drawdown analysis
            cumulative_returns = (1 + returns_series/100).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max * 100
            max_drawdown = drawdowns.min()
            
            # Trading statistics
            number_of_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            win_rate = (winning_trades / number_of_trades * 100) if number_of_trades > 0 else 0
            
            # Profit factor
            gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum()
            gross_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Consecutive losses
            trade_results = trades_df['profit'].apply(lambda x: 'win' if x > 0 else 'loss')
            consecutive_losses = 0
            max_consecutive_losses = 0
            for result in trade_results:
                if result == 'loss':
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
                    
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'number_of_trades': number_of_trades,
                'average_return': trades_df['return'].mean(),
                'max_consecutive_losses': max_consecutive_losses,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'average_trade_duration': (trades_df['exit_date'] - trades_df['entry_date']).mean().days,
                'profit_per_trade': trades_df['profit'].mean(),
                'best_trade': trades_df['return'].max(),
                'worst_trade': trades_df['return'].min(),
                'final_capital': final_capital
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise