import torch
import numpy as np
from data.data_loader import DataLoader
from models.torch_net import TorchNet
import os
import time
from datetime import datetime
import schedule
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)

class LiveTrader:
    def __init__(self, symbols=None, initial_capital=1000):
        if symbols is None:
            self.symbols = [
                'BTC/USDT',
                'BNB/USDT',
                'LTC/USDT',
                'LINK/USDT',
                'NEAR/USDT',
                'ETH/USDT',
                'DOGE/USDT'
            ]
        else:
            self.symbols = symbols
            
        self.traders = {}
        for symbol in self.symbols:
            self.traders[symbol] = {
                'model': self._load_model(symbol),
                'data_loader': DataLoader(data_source='binance', symbol=symbol)
            }
    
    def _load_model(self, symbol):
        model = TorchNet()
        model_filename = f'models/{symbol.replace("/", "_")}_model.pth'
        logging.info(f"Loading model from {model_filename}")
        
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"No trained model found for {symbol}")
        
        model.load_state_dict(torch.load(model_filename))
        model.eval()
        return model
    
    def get_all_signals(self):
        signals = {}
        try:
            for symbol in self.symbols:
                # Get latest market data
                market_data = self.traders[symbol]['data_loader'].load_data(
                    use_cache=False, 
                    write_cache=False
                )
                sequence_length = 99
                
                # Prepare latest sequence
                sequence = market_data[['open', 'high', 'low', 'close', 'volume']].values[-sequence_length:]
                sequence = torch.FloatTensor(sequence).unsqueeze(0)
                
                # Get model prediction
                with torch.no_grad():
                    signal = self.traders[symbol]['model'](sequence).item()
                    signal = np.clip(signal, -1, 1)
                
                current_price = market_data['close'].values[-1]
                signals[symbol] = {
                    'signal': signal,
                    'price': current_price,
                    'action': 'BUY' if signal > 0 else 'SELL' if signal < 0 else 'HOLD'
                }
                
        except Exception as e:
            logging.error(f"Error getting signals: {str(e)}")
        
        return signals

def run_strategy():
    trader = LiveTrader()
    signals = trader.get_all_signals()
    
    logging.info("\n=== Strategy Update ===")
    logging.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for symbol, data in signals.items():
        logging.info(f"\n{symbol}:")
        logging.info(f"Current Price: ${data['price']:.2f}")
        logging.info(f"Strategy Signal: {data['signal']:.4f}")
        logging.info(f"Recommended Action: {data['action']}")

def main():
    logging.info("Starting multi-symbol trading strategy...")
    schedule.every(4).hours.do(run_strategy)
    
    # Run immediately first time
    run_strategy()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == '__main__':
    main()