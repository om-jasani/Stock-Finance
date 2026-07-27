"""Strategy optimization utilities"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Type, Tuple
import itertools
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import spearmanr
from .backtester import Backtester
from .strategies import TradingStrategy
from .logger import logger
from .exceptions import ValidationError

class StrategyOptimizer:
    """Class for optimizing strategy parameters"""
    
    def __init__(self, 
                strategy_class: Type[TradingStrategy],
                param_grid: Dict[str, List[Any]],
                initial_capital: float = 100000):
        """
        Initialize optimizer
        
        Args:
            strategy_class: Trading strategy class to optimize
            param_grid: Dictionary of parameter names and possible values
            initial_capital: Initial capital for backtesting
        """
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.backtester = Backtester(initial_capital=initial_capital)
        
    def optimize(self,
                symbol: str,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                metric: str = 'sharpe_ratio',
                max_workers: int = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Find optimal strategy parameters
        
        Args:
            symbol: Stock symbol
            start_date: Start date for optimization
            end_date: End date for optimization
            metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            max_workers: Maximum number of parallel workers
            
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Best parameters and results
        """
        try:
            # Generate parameter combinations
            param_combinations = [dict(zip(self.param_grid.keys(), v)) 
                               for v in itertools.product(*self.param_grid.values())]
            
            logger.info(f"Testing {len(param_combinations)} parameter combinations")
            
            # Function to test a single parameter combination
            def test_parameters(params):
                try:
                    strategy = self.strategy_class(params)
                    results = self.backtester.run_backtest(
                        symbol=symbol,
                        strategy=strategy.generate_signal,
                        start_date=start_date,
                        end_date=end_date
                    )
                    return params, results.metrics[metric], results.metrics
                except Exception as e:
                    logger.error(f"Error testing parameters {params}: {str(e)}")
                    return params, float('-inf'), {}
            
            # Run optimization in parallel
            results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(test_parameters, params) 
                          for params in param_combinations]
                
                for future in futures:
                    params, score, metrics = future.result()
                    results.append((params, score, metrics))
            
            # Find best parameters
            best_params, best_score, best_metrics = max(results, key=lambda x: x[1])
            
            # Calculate parameter importance and distributions
            param_importance = self._calculate_parameter_importance(results)
            param_distributions = self._analyze_parameter_distributions(results)
            
            # Create optimization summary
            summary = {
                'tested_combinations': len(param_combinations),
                'optimization_metric': metric,
                'best_score': best_score,
                'parameter_importance': param_importance,
                'parameter_distributions': param_distributions,
                'top_results': self._get_top_results(results, n=5),
                'metric_statistics': self._calculate_metric_statistics(results),
                'robustness_score': self._calculate_robustness_score(results)
            }
            
            return best_params, summary
            
        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}")
            raise
            
    def _calculate_parameter_importance(self, results: List[Tuple[Dict[str, Any], float, Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate parameter importance based on correlation with performance"""
        try:
            importance = {}
            scores = np.array([score for _, score, _ in results])
            
            for param_name in self.param_grid.keys():
                param_values = np.array([res[0][param_name] for res in results])
                if isinstance(param_values[0], (int, float)):
                    corr, _ = spearmanr(param_values, scores)
                    importance[param_name] = abs(corr)
                else:
                    # For categorical parameters, calculate variance in performance
                    unique_values = set(param_values)
                    value_scores = {val: [] for val in unique_values}
                    for val, score in zip(param_values, scores):
                        value_scores[val].append(score)
                    variances = [np.var(scores) for scores in value_scores.values()]
                    importance[param_name] = np.mean(variances)
            
            # Normalize importance scores
            max_importance = max(importance.values())
            importance = {k: v/max_importance for k, v in importance.items()}
            
            return importance
            
    def _analyze_parameter_distributions(self, results: List[Tuple[Dict[str, Any], float, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Analyze parameter value distributions in top-performing results"""
        try:
            # Sort results by score
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            top_n = int(len(results) * 0.1)  # Top 10%
            top_results = sorted_results[:top_n]
            
            distributions = {}
            for param_name in self.param_grid.keys():
                param_values = [res[0][param_name] for res in top_results]
                
                if isinstance(param_values[0], (int, float)):
                    distributions[param_name] = {
                        'mean': np.mean(param_values),
                        'std': np.std(param_values),
                        'median': np.median(param_values),
                        'min': min(param_values),
                        'max': max(param_values)
                    }
                else:
                    # For categorical parameters, calculate frequency
                    value_counts = pd.Series(param_values).value_counts()
                    distributions[param_name] = {
                        'mode': value_counts.index[0],
                        'frequencies': value_counts.to_dict()
                    }
            
            return distributions
            
    def _get_top_results(self, results: List[Tuple[Dict[str, Any], float, Dict[str, Any]]], n: int = 5) -> List[Dict[str, Any]]:
        """Get top N performing parameter combinations"""
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        
        top_results = []
        for params, score, metrics in sorted_results[:n]:
            top_results.append({
                'parameters': params,
                'score': score,
                'key_metrics': {
                    'sharpe_ratio': metrics.get('sharpe_ratio'),
                    'total_return': metrics.get('total_return'),
                    'max_drawdown': metrics.get('max_drawdown'),
                    'win_rate': metrics.get('win_rate')
                }
            })
            
        return top_results
        
    def _calculate_metric_statistics(self, results: List[Tuple[Dict[str, Any], float, Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for various performance metrics"""
        metrics = {}
        for key in ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']:
            values = [res[2].get(key, 0) for res in results]
            metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': min(values),
                'max': max(values),
                'skew': pd.Series(values).skew(),
                'kurtosis': pd.Series(values).kurtosis()
            }
        return metrics
        
    def _calculate_robustness_score(self, results: List[Tuple[Dict[str, Any], float, Dict[str, Any]]]) -> float:
        """Calculate strategy robustness score"""
        try:
            scores = np.array([score for _, score, _ in results])
            
            # Calculate various robustness metrics
            consistency = len(scores[scores > 0]) / len(scores)  # Proportion of positive results
            stability = 1 - (np.std(scores) / np.mean(scores))  # Lower variation is better
            worst_case = np.percentile(scores, 5) / np.median(scores)  # Tail risk
            
            # Combine metrics into final score
            robustness_score = (consistency * 0.4 + 
                              stability * 0.4 + 
                              worst_case * 0.2)
            
            return max(0, min(1, robustness_score))  # Normalize to [0, 1]
            
        except Exception as e:
            logger.error(f"Error calculating robustness score: {str(e)}")
            return 0.0

class WalkForwardOptimizer(StrategyOptimizer):
    """Walk-forward optimization for trading strategies"""
    
    def optimize(self,
               symbol: str,
               start_date: Optional[str] = None,
               end_date: Optional[str] = None,
               train_size: int = 252,  # 1 year of trading days
               test_size: int = 63,    # 3 months of trading days
               step_size: int = 21,    # 1 month steps
               metric: str = 'sharpe_ratio',
               max_workers: int = None) -> Dict[str, Any]:
        """
        Perform walk-forward optimization
        
        Args:
            symbol: Stock symbol
            start_date: Start date for optimization
            end_date: End date for optimization
            train_size: Number of days in training window
            test_size: Number of days in test window
            step_size: Number of days to step forward
            metric: Metric to optimize
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        try:
            # Get historical data
            data = self.backtester.data_fetcher.get_stock_data(
                symbol, start_date=start_date, end_date=end_date
            )
            
            if data is None or len(data) < train_size + test_size:
                raise ValidationError("Insufficient data for walk-forward optimization")
                
            # Initialize results storage
            walk_forward_results = []
            
            # Walk through the data
            for i in range(0, len(data) - train_size - test_size, step_size):
                # Split data into train and test periods
                train_data = data.iloc[i:i+train_size]
                test_data = data.iloc[i+train_size:i+train_size+test_size]
                
                # Optimize parameters on training data
                train_start = train_data.index[0].strftime('%Y-%m-%d')
                train_end = train_data.index[-1].strftime('%Y-%m-%d')
                
                best_params, _ = super().optimize(
                    symbol=symbol,
                    start_date=train_start,
                    end_date=train_end,
                    metric=metric,
                    max_workers=max_workers
                )
                
                # Test parameters on out-of-sample data
                test_start = test_data.index[0].strftime('%Y-%m-%d')
                test_end = test_data.index[-1].strftime('%Y-%m-%d')
                
                strategy = self.strategy_class(best_params)
                test_results = self.backtester.run_backtest(
                    symbol=symbol,
                    strategy=strategy.generate_signal,
                    start_date=test_start,
                    end_date=test_end
                )
                
                walk_forward_results.append({
                    'train_period': (train_start, train_end),
                    'test_period': (test_start, test_end),
                    'parameters': best_params,
                    'train_score': test_results.metrics[metric],
                    'test_metrics': test_results.metrics
                })
            
            # Analyze walk-forward results
            analysis = self._analyze_walk_forward_results(walk_forward_results)
            
            return {
                'walk_forward_results': walk_forward_results,
                'analysis': analysis,
                'recommended_parameters': self._get_recommended_parameters(walk_forward_results)
            }
            
        except Exception as e:
            logger.error(f"Error in walk-forward optimization: {str(e)}")
            raise
            
    def _analyze_walk_forward_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze walk-forward optimization results"""
        try:
            # Calculate parameter stability
            param_stability = {}
            for param in self.param_grid.keys():
                values = [res['parameters'][param] for res in results]
                param_stability[param] = np.std(values) / np.mean(values)
            
            # Calculate performance consistency
            train_scores = [res['train_score'] for res in results]
            test_scores = [res['test_metrics'].get('sharpe_ratio', 0) for res in results]
            
            return {
                'parameter_stability': param_stability,
                'avg_train_score': np.mean(train_scores),
                'avg_test_score': np.mean(test_scores),
                'train_test_correlation': np.corrcoef(train_scores, test_scores)[0, 1],
                'optimization_decay': np.mean(np.array(test_scores) / np.array(train_scores))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing walk-forward results: {str(e)}")
            raise
            
    def _get_recommended_parameters(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get recommended parameters based on walk-forward results"""
        try:
            # Weight recent results more heavily
            weights = np.linspace(0.5, 1.0, len(results))
            recommended_params = {}
            
            for param in self.param_grid.keys():
                values = [res['parameters'][param] for res in results]
                if isinstance(values[0], (int, float)):
                    # For numeric parameters, use weighted average
                    recommended_params[param] = np.average(values, weights=weights)
                else:
                    # For categorical parameters, use weighted mode
                    unique_values = set(values)
                    weighted_counts = {val: 0 for val in unique_values}
                    for val, weight in zip(values, weights):
                        weighted_counts[val] += weight
                    recommended_params[param] = max(weighted_counts.items(), key=lambda x: x[1])[0]
            
            return recommended_params
            
        except Exception as e:
            logger.error(f"Error getting recommended parameters: {str(e)}")
            raise