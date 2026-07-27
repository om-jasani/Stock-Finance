"""Risk management utilities"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize
from .data_fetcher import DataFetcher
from .exceptions import ValidationError
from .logger import logger

class RiskManager:
    """Class for managing portfolio risk and optimization"""
    
    def __init__(self):
        """Initialize RiskManager"""
        self.data_fetcher = DataFetcher()
        
    def calculate_portfolio_metrics(self, 
                                 positions: Dict[str, Dict[str, float]],
                                 risk_free_rate: float = 0.04) -> Dict[str, Any]:
        """
        Calculate portfolio risk metrics
        
        Args:
            positions: Dictionary of positions {symbol: {'shares': shares, 'purchase_price': price}}
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dict[str, Any]: Portfolio metrics
        """
        try:
            # Fetch historical data for all symbols
            symbols = list(positions.keys())
            historical_data = self.data_fetcher.get_multiple_stocks(symbols, period='1y')
            
            if not historical_data:
                raise ValidationError("No historical data available")
                
            # Calculate returns and weights
            returns_data = {}
            weights = []
            total_value = 0
            
            for symbol, position in positions.items():
                if symbol in historical_data:
                    # Calculate position value
                    current_price = historical_data[symbol]['Close'].iloc[-1]
                    position_value = position['shares'] * current_price
                    total_value += position_value
                    
                    # Store returns data
                    returns_data[symbol] = historical_data[symbol]['Close'].pct_change()
                    weights.append(position_value)
            
            if total_value == 0:
                raise ValidationError("Portfolio has no value")
                
            # Normalize weights
            weights = np.array(weights) / total_value
            
            # Create returns matrix
            returns_matrix = pd.DataFrame(returns_data)
            
            # Calculate portfolio metrics
            portfolio_return = self._calculate_portfolio_return(weights, returns_matrix)
            portfolio_volatility = self._calculate_portfolio_volatility(weights, returns_matrix)
            
            # Calculate additional metrics
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            # Calculate Value at Risk (VaR)
            var_95 = self._calculate_var(weights, returns_matrix, confidence_level=0.95)
            var_99 = self._calculate_var(weights, returns_matrix, confidence_level=0.99)
            
            # Calculate Conditional Value at Risk (CVaR)
            cvar_95 = self._calculate_cvar(weights, returns_matrix, confidence_level=0.95)
            
            # Calculate diversification metrics
            correlation_matrix = returns_matrix.corr()
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            
            return {
                'total_value': total_value,
                'annual_return': portfolio_return * 252,  # Annualized
                'annual_volatility': portfolio_volatility * np.sqrt(252),  # Annualized
                'sharpe_ratio': sharpe_ratio * np.sqrt(252),  # Annualized
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'avg_correlation': avg_correlation,
                'weights': dict(zip(symbols, weights)),
                'risk_contribution': self._calculate_risk_contribution(weights, returns_matrix)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            raise
            
    def optimize_portfolio(self,
                         symbols: List[str],
                         target_return: Optional[float] = None,
                         risk_free_rate: float = 0.04,
                         optimization_goal: str = 'sharpe') -> Dict[str, Any]:
        """
        Optimize portfolio weights
        
        Args:
            symbols: List of stock symbols
            target_return: Target portfolio return (optional)
            risk_free_rate: Risk-free rate
            optimization_goal: 'sharpe', 'min_risk', or 'risk_parity'
            
        Returns:
            Dict[str, Any]: Optimized portfolio weights and metrics
        """
        try:
            # Fetch historical data
            historical_data = self.data_fetcher.get_multiple_stocks(symbols, period='1y')
            
            if not historical_data:
                raise ValidationError("No historical data available")
                
            # Create returns matrix
            returns_data = {}
            for symbol in symbols:
                if symbol in historical_data:
                    returns_data[symbol] = historical_data[symbol]['Close'].pct_change()
            
            returns_matrix = pd.DataFrame(returns_data)
            
            # Define optimization constraints
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # weights sum to 1
            bounds = tuple((0, 1) for _ in range(len(symbols)))  # weights between 0 and 1
            
            # Initial guess (equal weights)
            initial_weights = np.array([1/len(symbols)] * len(symbols))
            
            if optimization_goal == 'sharpe':
                # Maximize Sharpe ratio
                objective = lambda w: -self._calculate_sharpe_ratio(w, returns_matrix, risk_free_rate)
            elif optimization_goal == 'min_risk':
                # Minimize portfolio volatility
                objective = lambda w: self._calculate_portfolio_volatility(w, returns_matrix)
                if target_return is not None:
                    constraints.append({
                        'type': 'eq',
                        'fun': lambda w: self._calculate_portfolio_return(w, returns_matrix) - target_return
                    })
            elif optimization_goal == 'risk_parity':
                # Risk parity portfolio
                objective = lambda w: self._calculate_risk_parity_objective(w, returns_matrix)
            else:
                raise ValidationError(f"Unknown optimization goal: {optimization_goal}")
            
            # Run optimization
            result = minimize(objective,
                           initial_weights,
                           method='SLSQP',
                           bounds=bounds,
                           constraints=constraints)
            
            if not result.success:
                raise ValidationError(f"Optimization failed: {result.message}")
                
            # Calculate metrics for optimized portfolio
            optimized_weights = result.x
            metrics = {
                'weights': dict(zip(symbols, optimized_weights)),
                'annual_return': self._calculate_portfolio_return(optimized_weights, returns_matrix) * 252,
                'annual_volatility': self._calculate_portfolio_volatility(optimized_weights, returns_matrix) * np.sqrt(252),
                'sharpe_ratio': self._calculate_sharpe_ratio(optimized_weights, returns_matrix, risk_free_rate) * np.sqrt(252)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            raise
            
    def _calculate_portfolio_return(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """Calculate portfolio return"""
        return np.sum(returns.mean() * weights)
        
    def _calculate_portfolio_volatility(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
        
    def _calculate_sharpe_ratio(self, weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        portfolio_ret = self._calculate_portfolio_return(weights, returns)
        portfolio_vol = self._calculate_portfolio_volatility(weights, returns)
        return (portfolio_ret - risk_free_rate) / portfolio_vol
        
    def _calculate_var(self, weights: np.ndarray, returns: pd.DataFrame, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        portfolio_returns = np.sum(returns * weights, axis=1)
        return np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
    def _calculate_cvar(self, weights: np.ndarray, returns: pd.DataFrame, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk"""
        portfolio_returns = np.sum(returns * weights, axis=1)
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return portfolio_returns[portfolio_returns <= var].mean()
        
    def _calculate_risk_contribution(self, weights: np.ndarray, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk contribution of each asset"""
        portfolio_vol = self._calculate_portfolio_volatility(weights, returns)
        risk_contribution = {}
        
        for i, symbol in enumerate(returns.columns):
            component_vol = weights[i] * np.dot(returns.cov().iloc[i], weights) / portfolio_vol
            risk_contribution[symbol] = component_vol / portfolio_vol
            
        return risk_contribution
        
    def _calculate_risk_parity_objective(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """Objective function for risk parity optimization"""
        risk_contributions = []
        portfolio_vol = self._calculate_portfolio_volatility(weights, returns)
        
        for i in range(len(weights)):
            component_vol = weights[i] * np.dot(returns.cov().iloc[i], weights) / portfolio_vol
            risk_contributions.append(component_vol)
            
        risk_contributions = np.array(risk_contributions)
        return np.sum((risk_contributions - risk_contributions.mean())**2)