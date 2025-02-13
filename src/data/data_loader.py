import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from sklearn.preprocessing import MinMaxScaler
import numpy as np

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_source, symbol='ETH/USDT', timeframe='4h'):
        """
        Initialize DataLoader with exchange and trading pair
        
        Args:
            data_source (str): Exchange name (e.g., 'binance', 'coinbase')
            symbol (str): Trading pair symbol
            timeframe (str): Candlestick timeframe
        """
        self.exchange = getattr(ccxt, data_source)()
        self.symbol = symbol
        self.timeframe = timeframe
        # Create cache directory if it doesn't exist
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_filename(self):
        """Generate cache filename based on parameters"""
        symbol_clean = self.symbol.replace('/', '_')
        return os.path.join(self.cache_dir, f"{symbol_clean}_{self.timeframe}_data.csv")

    def _save_to_cache(self, df):
        """Save DataFrame to local cache"""
        try:
            cache_file = self._get_cache_filename()
            df.to_csv(cache_file)
            logger.info(f"Data saved to cache: {cache_file}")
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")

    def _load_from_cache(self):
        """Load DataFrame from local cache if available"""
        try:
            cache_file = self._get_cache_filename()
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col=0)
                df.index = pd.to_datetime(df.index)
                logger.info(f"Data loaded from cache: {cache_file}")
                return df
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
        return None

    def load_data(self, start_date=None, limit=15000, use_cache=True, normalize=True):
        """
        Load historical market data with caching support and optional normalization
        
        Args:
            start_date (datetime): Start date for historical data
            limit (int): Total number of candles to fetch
            use_cache (bool): Whether to use cached data if available
            normalize (bool): Whether to normalize the data
            
        Returns:
            pd.DataFrame: Historical market data (normalized if specified)
        """
        if use_cache:
            cached_data = self._load_from_cache()
            if cached_data is not None:
                return cached_data

        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=limit/6)
            
            all_ohlcv = []
            batch_size = 1000
            remaining = limit
            last_timestamp = int(start_date.timestamp() * 1000)

            while remaining > 0:
                batch = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=last_timestamp,
                    limit=min(batch_size, remaining)
                )
                
                if not batch:
                    break
                    
                all_ohlcv.extend(batch)
                remaining -= len(batch)
                
                # Update timestamp for next iteration
                last_timestamp = batch[-1][0] + 1
                
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.drop_duplicates()  # Remove any potential duplicates

            # Save to cache
            
            
            if normalize and df is not None:
                df = self.normalize_data(df)
            df['delta']=df['close'].diff()
            self._save_to_cache(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def normalize_data(self, df):
        """
        Normalize the OHLCV data using MinMaxScaler
        
        Args:
            df (pd.DataFrame): Original DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        try:
            # Create a copy of the DataFrame
            df_normalized = df.copy()
            
            # Initialize the scaler
            scaler = MinMaxScaler()
            
            # Columns to normalize
            columns_to_normalize = ['open', 'high', 'low', 'close', 'volume']
            
            # Fit and transform the data
            normalized_data = scaler.fit_transform(df_normalized[columns_to_normalize])
            
            # Update the DataFrame with normalized values
            df_normalized[columns_to_normalize] = normalized_data
            
            return df_normalized
            
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return df

    def get_latest_data(self):
        """
        Get the most recent market data
        
        Returns:
            dict: Latest market data including price and volume
        """
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            
            return {
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000),
                'last_price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'change_24h': ticker['percentage']
            }
            
        except Exception as e:
            print(f"Error fetching latest data: {e}")
            return None