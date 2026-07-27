"""Data caching system"""
import pandas as pd
from typing import Optional, Dict, Any
import json
import os
from datetime import datetime, timedelta
from .logger import logger
from .config import Config

class DataCache:
    """Simple data caching system"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize cache
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir or os.path.join(Config.DATA_DIR, 'cache')
        self.cache_duration = timedelta(minutes=Config.DATA_CACHE_MINUTES)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load cache metadata
        self.metadata_file = os.path.join(self.cache_dir, 'metadata.json')
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading cache metadata: {str(e)}")
            return {}
            
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {str(e)}")
            
    def _get_cache_key(self, symbol: str, period: str, interval: str) -> str:
        """Generate cache key"""
        return f"{symbol}_{period}_{interval}"
        
    def _get_cache_file(self, key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{key}.parquet")
        
    def get(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Get data from cache
        
        Args:
            symbol: Stock symbol
            period: Time period
            interval: Data interval
            
        Returns:
            Optional[pd.DataFrame]: Cached data if available and valid
        """
        try:
            key = self._get_cache_key(symbol, period, interval)
            cache_file = self._get_cache_file(key)
            
            # Check if cache exists and is valid
            if (key in self.metadata and 
                os.path.exists(cache_file) and
                datetime.now() - datetime.fromisoformat(self.metadata[key]['timestamp']) < self.cache_duration):
                
                return pd.read_parquet(cache_file)
                
            return None
            
        except Exception as e:
            logger.error(f"Error reading from cache: {str(e)}")
            return None
            
    def set(self, symbol: str, period: str, interval: str, data: pd.DataFrame):
        """
        Store data in cache
        
        Args:
            symbol: Stock symbol
            period: Time period
            interval: Data interval
            data: Data to cache
        """
        try:
            key = self._get_cache_key(symbol, period, interval)
            cache_file = self._get_cache_file(key)
            
            # Save data
            data.to_parquet(cache_file)
            
            # Update metadata
            self.metadata[key] = {
                'timestamp': datetime.now().isoformat(),
                'rows': len(data),
                'period': period,
                'interval': interval,
                'symbol': symbol,
                'columns': list(data.columns)
            }
            
            # Save metadata
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Error writing to cache: {str(e)}")
            
    def clear(self, symbol: Optional[str] = None):
        """
        Clear cache
        
        Args:
            symbol: Optional symbol to clear specific cache
        """
        try:
            if symbol:
                # Clear specific symbol
                keys_to_delete = [
                    key for key in self.metadata 
                    if key.startswith(symbol + '_')
                ]
                
                for key in keys_to_delete:
                    cache_file = self._get_cache_file(key)
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                    del self.metadata[key]
            else:
                # Clear all cache
                for key in self.metadata:
                    cache_file = self._get_cache_file(key)
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                self.metadata = {}
                
            # Save metadata
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            
    def cleanup(self):
        """Remove expired cache entries"""
        try:
            current_time = datetime.now()
            keys_to_delete = []
            
            # Find expired entries
            for key, info in self.metadata.items():
                cache_time = datetime.fromisoformat(info['timestamp'])
                if current_time - cache_time > self.cache_duration:
                    keys_to_delete.append(key)
                    
            # Remove expired entries
            for key in keys_to_delete:
                cache_file = self._get_cache_file(key)
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                del self.metadata[key]
                
            # Save metadata
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        try:
            total_size = 0
            for key in self.metadata:
                cache_file = self._get_cache_file(key)
                if os.path.exists(cache_file):
                    total_size += os.path.getsize(cache_file)
                    
            return {
                'entries': len(self.metadata),
                'total_size_mb': total_size / (1024 * 1024),
                'symbols': len(set(info['symbol'] for info in self.metadata.values())),
                'oldest_entry': min(info['timestamp'] for info in self.metadata.values()) if self.metadata else None,
                'newest_entry': max(info['timestamp'] for info in self.metadata.values()) if self.metadata else None
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}