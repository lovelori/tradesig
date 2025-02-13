import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from sklearn.preprocessing import MinMaxScaler
import numpy as np

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_source=None, symbol='ETH/USDT', timeframe='4h'):
        """
        Initialize DataLoader with trading pair
        
        Args:
            data_source: API key for CryptoCompare
            symbol (str): Trading pair symbol (e.g., 'ETH/USDT')
            timeframe (str): Candlestick timeframe
        """
        self.base_url = "https://data-api.cryptocompare.com/spot/v1/historical/hours?"
        self.symbol = symbol
        self.api_key = data_source  # Required API key
        # Convert symbol format from 'ETH/USDT' to 'ETH-USDT'
        self.instrument = symbol.replace('/', '-')
        # Map timeframe
        self.timeframe, self.aggregate = self._convert_timeframe(timeframe)
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _convert_timeframe(self, timeframe):
        """Convert timeframe to CryptoCompare format"""
        if timeframe.endswith('m'):
            return 'minutes', int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return 'hours', int(timeframe[:-1])
        else:
            return 'days', 1

    def load_data(self, start_date=None, limit=2000, use_cache=True, normalize=True, write_cache=True):
        """
        Load historical market data from CryptoCompare
        Handles pagination for requests exceeding 500 records
        """
        if use_cache:
            cached_data = self._load_from_cache()
            if cached_data is not None:
                return cached_data

        try:
            all_data = []
            remaining_limit = limit
            current_timestamp = int(datetime.now().timestamp())

            while remaining_limit > 0:
                # Calculate batch size (max 500)
                batch_size = min(500, remaining_limit)

                params = {
                    'market': 'binance',
                    'instrument': self.instrument,
                    'limit': batch_size,
                    'aggregate': self.aggregate,
                    'fill': 'true',
                    'apply_mapping': 'true',
                    'response_format': 'JSON',
                    'to_ts': current_timestamp
                }

                headers = {
                    'authorization': f"Bearer {self.api_key}"
                }

                response = requests.get(self.base_url, params=params, headers=headers)
                if response.status_code != 200:
                    logger.error(f"API request failed: {response.status_code} - {response.text}")
                    break

                data = response.json()
                
                batch_data = data['Data']
                all_data.extend(batch_data)

                # Update timestamp for next batch
                if batch_data:
                    current_timestamp = int(batch_data[0]["TIMESTAMP"]) - 1
                    print(batch_data[-1]["TIMESTAMP"])
                else:
                    break

                remaining_limit -= batch_size
                
                # Add delay to avoid rate limiting

            if not all_data:
                logger.error("No data collected from API")
                return None
            
            # Convert all collected data to DataFrame
            try:
                df = pd.DataFrame(all_data)
                #df['timestamp'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
                df = df.rename(columns={
                    'TIMESTAMP':'timestamp',
                    'OPEN': 'open',
                    'HIGH': 'high',
                    'LOW': 'low',
                    'CLOSE': 'close',
                    'VOLUME': 'volume'
                })

                # Sort by timestamp and remove duplicates
                df = df.sort_values('timestamp')
                df = df.drop_duplicates(subset=['timestamp'])
                
                # Clean up and set index
                df = df.set_index('timestamp')
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df = df.astype(float)

                if normalize:
                    df = self.normalize_data(df)
                
                df['delta'] = df['close'].diff()
                
                if write_cache:
                    self._save_to_cache(df)
                
                return df

            except Exception as e:
                logger.error(f"Error processing DataFrame: {e}")
                return None

        except Exception as e:
            logger.error(f"Error loading data from CryptoCompare: {e}")
            return None

    def get_latest_data(self):
        """Get the most recent market data"""
        try:
            url = f"{self.base_url}/price"
            params = {
                'fsym': self.base_currency,
                'tsyms': self.quote_currency,
                'e': 'binance'
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                return None
                
            price_data = response.json()
            
            # Get 24h data for additional info
            url_24h = f"{self.base_url}/v2/histohour"
            params_24h = {
                'fsym': self.base_currency,
                'tsym': self.quote_currency,
                'limit': 24,
                'e': 'binance'
            }
            
            response_24h = requests.get(url_24h, params=params_24h)
            data_24h = response_24h.json()['Data']['Data']
            
            # Calculate 24h change
            price_24h_ago = data_24h[0]['close']
            current_price = price_data[self.quote_currency]
            change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
            
            # Calculate 24h volume
            volume_24h = sum(x['volumefrom'] for x in data_24h)
            
            return {
                'timestamp': datetime.now(),
                'last_price': float(current_price),
                'volume': float(volume_24h),
                'change_24h': float(change_24h)
            }
            
        except Exception as e:
            logger.error(f"Error fetching latest data: {e}")
            return None

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