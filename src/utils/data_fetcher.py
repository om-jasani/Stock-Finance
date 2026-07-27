"""Stock data fetching utility"""
import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from .cache import DataCache
from .exceptions import DataFetchError
from .logger import logger
from .validation import validate_stock_data
from .config import Config

# Approximate day-count for period strings, used only to size fallback
# provider date ranges (yfinance's own period parsing is unaffected)
_PERIOD_DAYS = {
    '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180,
    '1y': 365, '2y': 730, '5y': 1825, '10y': 3650, 'max': 3650
}

class DataFetcher:
    """Utility class for fetching stock market data"""

    def __init__(self):
        """Initialize DataFetcher"""
        self.cache = DataCache()
        self.batch_size = 5  # Number of symbols to fetch in parallel

    def get_stock_data(self,
                      symbol: str,
                      period: str = '1d',
                      interval: str = '5m',
                      use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch stock data for given symbol. Falls back to Alpha Vantage, then
        Finnhub, for daily-interval requests if yfinance fails or rate-limits
        and a corresponding API key is configured - yfinance is convenient
        but well known to break/rate-limit intermittently.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            use_cache: Whether to use cached data

        Returns:
            Optional[pd.DataFrame]: DataFrame with stock data or None if fetch failed
        """
        try:
            # Check cache first
            if use_cache:
                cached_data = self.cache.get(symbol, period, interval)
                if cached_data is not None:
                    logger.debug(f"Retrieved {symbol} data from cache")
                    return cached_data

            df = self._fetch_with_fallback(symbol, period, interval)

            if df is None or df.empty:
                logger.warning(f"No data received for symbol {symbol}")
                return None

            # Validate data
            validate_stock_data(df)

            # Add technical indicators
            df = self._add_technical_indicators(df)

            # Cache the result
            if use_cache:
                self.cache.set(symbol, period, interval, df)

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise DataFetchError(f"Failed to fetch data for {symbol}: {str(e)}")

    def _fetch_with_fallback(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Try yfinance first, then Alpha Vantage/Finnhub for daily data if it fails"""
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval)
            if not df.empty:
                return df
            logger.warning(f"yfinance returned no data for {symbol}, trying fallback providers")
        except Exception as e:
            logger.warning(f"yfinance fetch failed for {symbol} ({e}), trying fallback providers")

        # The free-tier fallback providers only make sense for daily bars;
        # their intraday history/entitlements are far more limited than yfinance's.
        if interval not in ('1d', '1D', 'daily'):
            return None

        for fetch_fn, name in (
            (self._fetch_alpha_vantage, 'Alpha Vantage'),
            (self._fetch_finnhub, 'Finnhub'),
        ):
            try:
                df = fetch_fn(symbol, period)
                if df is not None and not df.empty:
                    logger.info(f"Fetched {symbol} from {name} fallback")
                    return df
            except Exception as e:
                logger.warning(f"{name} fallback failed for {symbol}: {e}")

        return None

    def _fetch_alpha_vantage(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fallback fetch via Alpha Vantage's daily time series"""
        if not Config.ALPHA_VANTAGE_API_KEY:
            return None

        from alpha_vantage.timeseries import TimeSeries

        ts = TimeSeries(key=Config.ALPHA_VANTAGE_API_KEY, output_format='pandas')
        outputsize = 'full' if _PERIOD_DAYS.get(period, 30) > 100 else 'compact'
        data, _ = ts.get_daily(symbol, outputsize=outputsize)

        data = data.rename(columns={
            '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
            '4. close': 'Close', '5. volume': 'Volume'
        })
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        cutoff = datetime.now() - timedelta(days=_PERIOD_DAYS.get(period, 30))
        return data[data.index >= cutoff]

    def _fetch_finnhub(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fallback fetch via Finnhub's daily candles"""
        if not Config.FINNHUB_API_KEY:
            return None

        import finnhub

        client = finnhub.Client(api_key=Config.FINNHUB_API_KEY)
        end = datetime.now()
        start = end - timedelta(days=_PERIOD_DAYS.get(period, 30))
        candles = client.stock_candles(symbol, 'D', int(start.timestamp()), int(end.timestamp()))

        if candles.get('s') != 'ok' or not candles.get('t'):
            return None

        df = pd.DataFrame({
            'Open': candles['o'], 'High': candles['h'], 'Low': candles['l'],
            'Close': candles['c'], 'Volume': candles['v']
        }, index=[datetime.fromtimestamp(t) for t in candles['t']])
        return df.sort_index()
            
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        try:
            df = df.copy()
            
            # Moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            
            # Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
            df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
            
            # RSI (guard divide-by-zero when there are no losses in the window)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            flat = (gain == 0) & (loss == 0)
            df['RSI'] = (100 - (100 / (1 + rs))).fillna(100).where(~flat, 50)
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Stochastic Oscillator
            low_min = df['Low'].rolling(window=14).min()
            high_max = df['High'].rolling(window=14).max()
            df['K_Line'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
            df['D_Line'] = df['K_Line'].rolling(window=3).mean()
            
            # Average True Range (ATR)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
            # Momentum
            df['Momentum'] = df['Close'].diff(periods=10)
            
            # Rate of Change (ROC)
            df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
            
            # Williams %R
            df['Williams_R'] = ((high_max - df['Close']) / (high_max - low_min)) * -100
            
            # On-Balance Volume (OBV)
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise
            
    def get_multiple_stocks(self, 
                          symbols: list, 
                          period: str = '1d', 
                          interval: str = '5m',
                          use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks in parallel"""
        try:
            results = {}
            failed_symbols = []
            
            def fetch_symbol(symbol):
                try:
                    data = self.get_stock_data(symbol, period, interval, use_cache)
                    return symbol, data
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {str(e)}")
                    return symbol, None
                    
            with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                future_to_symbol = {
                    executor.submit(fetch_symbol, symbol): symbol 
                    for symbol in symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol, data = future.result()
                    if data is not None:
                        results[symbol] = data
                    else:
                        failed_symbols.append(symbol)
                        
            if failed_symbols:
                logger.warning(f"Failed to fetch data for symbols: {failed_symbols}")
                
            return results
            
        except Exception as e:
            logger.error(f"Error fetching multiple stocks: {str(e)}")
            raise DataFetchError(f"Failed to fetch multiple stocks: {str(e)}")

    def get_sector_performance(self) -> Optional[pd.DataFrame]:
        """Get sector performance data"""
        try:
            # List of major sector ETFs
            sector_etfs = {
                'XLK': 'Technology',
                'XLF': 'Financials',
                'XLV': 'Healthcare',
                'XLE': 'Energy',
                'XLI': 'Industrials',
                'XLC': 'Communication Services',
                'XLP': 'Consumer Staples',
                'XLY': 'Consumer Discretionary',
                'XLB': 'Materials',
                'XLU': 'Utilities',
                'XLRE': 'Real Estate'
            }
            
            # Fetch data for all sectors
            sector_data = self.get_multiple_stocks(list(sector_etfs.keys()), period='1mo')
            
            # Calculate performance metrics
            performance = []
            for symbol, data in sector_data.items():
                if data is not None:
                    first_price = data['Close'].iloc[0]
                    last_price = data['Close'].iloc[-1]
                    change_pct = ((last_price - first_price) / first_price) * 100
                    
                    performance.append({
                        'Sector': sector_etfs[symbol],
                        'Symbol': symbol,
                        'Change_Pct': change_pct,
                        'Current_Price': last_price,
                        'Volume': data['Volume'].mean(),
                        'Volatility': data['Close'].pct_change().std() * 100
                    })
                    
            if not performance:
                return None
                
            return pd.DataFrame(performance)
            
        except Exception as e:
            logger.error(f"Error fetching sector performance: {str(e)}")
            return None
            
    def get_market_movers(self, n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get top gainers and losers"""
        try:
            # List of major stocks (you might want to expand this)
            major_stocks = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM',
                'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'ADBE', 'CRM',
                'NFLX', 'DIS', 'CSCO', 'VZ', 'INTC', 'KO', 'PEP', 'ABT', 'MRK',
                'PFE', 'NKE', 'TMO'
            ]
            
            # Fetch data
            stock_data = self.get_multiple_stocks(major_stocks, period='1d', interval='1m')
            
            # Calculate performance
            performance = []
            for symbol, data in stock_data.items():
                if data is not None:
                    first_price = data['Close'].iloc[0]
                    last_price = data['Close'].iloc[-1]
                    change_pct = ((last_price - first_price) / first_price) * 100
                    
                    performance.append({
                        'Symbol': symbol,
                        'Price': last_price,
                        'Change_Pct': change_pct,
                        'Volume': data['Volume'].sum()
                    })
                    
            if not performance:
                return pd.DataFrame(), pd.DataFrame()
                
            df = pd.DataFrame(performance)
            gainers = df.nlargest(n, 'Change_Pct')
            losers = df.nsmallest(n, 'Change_Pct')
            
            return gainers, losers
            
        except Exception as e:
            logger.error(f"Error fetching market movers: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
            
    def get_correlated_stocks(self, symbol: str, top_n: int = 5) -> Optional[pd.DataFrame]:
        """Find stocks with highest correlation to given symbol"""
        try:
            # Get main stock data
            main_data = self.get_stock_data(symbol, period='1y')
            if main_data is None:
                return None
                
            # List of stocks to compare (you might want to expand this)
            compare_stocks = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM',
                'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'ADBE', 'CRM'
            ]
            
            # Remove the input symbol from comparison list
            compare_stocks = [s for s in compare_stocks if s != symbol]
            
            # Get comparison data
            comparison_data = self.get_multiple_stocks(compare_stocks)
            
            # Calculate correlations
            correlations = []
            main_returns = main_data['Close'].pct_change()
            main_variance = main_returns.var()

            for comp_symbol, comp_data in comparison_data.items():
                if comp_data is not None:
                    comp_returns = comp_data['Close'].pct_change()
                    correlation = main_returns.corr(comp_returns)
                    beta = comp_returns.cov(main_returns) / main_variance if main_variance else np.nan

                    correlations.append({
                        'Symbol': comp_symbol,
                        'Correlation': correlation,
                        'Beta': beta
                    })
                    
            if not correlations:
                return None
                
            df = pd.DataFrame(correlations)
            return df.nlargest(top_n, 'Correlation')
            
        except Exception as e:
            logger.error(f"Error finding correlated stocks: {str(e)}")
            return None