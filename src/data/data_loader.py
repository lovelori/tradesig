import ccxt
import pandas as pd
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, data_source, symbol='BTC/USDT', timeframe='4h'):
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

    def load_data(self, start_date=None, limit=3000):
        """
        Load historical market data
        
        Args:
            start_date (datetime): Start date for historical data
            limit (int): Number of candles to fetch
            
        Returns:
            pd.DataFrame: Historical market data
        """
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            
            since = int(start_date.timestamp() * 1000)
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                since=since,
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

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