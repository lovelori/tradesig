import torch
import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import DataLoader
from models.torch_net import TorchNet,TorchNet2
from data.dataset import CryptoDataset
import os
import smtplib
import requests
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import base64
from io import BytesIO
import matplotlib
from requests.adapters import HTTPAdapter, Retry
matplotlib.use('Agg')  # Required for non-interactive backend

class Backtester:
    def __init__(self, initial_capital=1000):
        self.capital = initial_capital  # Cash
        self.position = 0  # Crypto holdings
        self.total_value_history = []
        self.price_history = []
        self.signal_history = []  # Add signal history

    def execute_trade(self, signal, current_price):
        """
        Execute trade based on model signal
        signal: float between -1 and 1
        """
        if current_price <= 0:
            return
        if signal > 0:  # Buy signal
            buy_amount = self.capital * abs(signal/2)
            self.position +=  0.9995*buy_amount / current_price
            self.capital -= buy_amount
        elif signal < 0:  # Sell signal
            sell_amount = self.position * abs(signal/2) 
            self.position -= sell_amount
            self.capital += 0.9995*sell_amount * current_price

    def get_total_value(self, current_price):
        return self.capital + (self.position * current_price)
def scan_binance_futures(min_multiplier=2.5, day_window=5, max_symbols=None):
    """Scan Binance Futures (USDT) perpetuals.
    Return list of dicts for symbols meeting: current_price > min_low*min_multiplier and latest fundingRate > 0.
    Robust to transient HTTP errors and common blocking (451).
    """
    exchange_info_url = "https://api.binance.com/api/v3/exchangeInfo"
    klines_url = "https://api.binance.com/api/v3/klines"
    price_url = "https://api.binance.com/api/v3/ticker/price"
    funding_url = "https://api.binance.com/api/v3/fundingRate"

    alerts = []

    # session with retries
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GitHub Actions/1.0)",
        "Accept": "application/json",
    }

    try:
        r = session.get(exchange_info_url, timeout=10, headers=headers)
        # handle legal/region block (451) or other non-200
        if r.status_code == 451:
            print("Received 451 from Binance API (likely region/legal block).")
            print("Options: run from a different runner/VPS, use a proxy, or use a fallback data provider.")
            return alerts
        r.raise_for_status()
        symbols = [s['symbol'] for s in r.json().get('symbols', []) if s.get('symbol','').endswith('USDT') and s.get('status') == 'TRADING']
    except requests.HTTPError as e:
        print(f"HTTP error fetching exchange info: {e} (status {getattr(e.response, 'status_code', 'N/A')})")
        return alerts
    except Exception as e:
        print(f"Failed to fetch exchange info: {e}")
        return alerts

    if max_symbols:
        symbols = symbols[:max_symbols]

    for symbol in symbols:
        try:
            params = {'symbol': symbol, 'interval': '1d', 'limit': day_window + 1}
            k = session.get(klines_url, params=params, timeout=8, headers=headers)
            k.raise_for_status()
            klines = k.json()
            if len(klines) < day_window:
                continue
            lows = [float(candle[3]) for candle in klines[-day_window:]]  # low is index 3
            min_low = min(lows)

            p = session.get(price_url, params={'symbol': symbol}, timeout=5, headers=headers)
            p.raise_for_status()
            price = float(p.json().get('price', 0.0))

            f = session.get(funding_url, params={'symbol': symbol, 'limit': 1}, timeout=6, headers=headers)
            f.raise_for_status()
            fr_list = f.json()
            funding_rate = float(fr_list[0].get('fundingRate', 0.0)) if fr_list else 0.0

            if price > min_low * min_multiplier and funding_rate > 0:
                alerts.append({
                    'symbol': symbol,
                    'price': price,
                    'min_low': min_low,
                    'multiplier': price / min_low if min_low > 0 else None,
                    'funding_rate': funding_rate
                })

            time.sleep(0.08)
        except Exception:
            # ignore symbol on any failure to keep scan resilient
            continue

    return alerts
def send_email(symbols_data, alerts=None):
    """Modified email function to include charts and optional alerts"""
    smtp_server = "smtp.qq.com"
    smtp_port = 465
    # Prefer providing these via environment variables (set from Actions secrets)
    sender_email = os.environ.get('SENDER_EMAIL', 'cdh40@qq.com')
    sender_password = os.environ.get('SENDER_PASSWORD')
    receiver_email = os.environ.get('RECEIVER_EMAIL', sender_email)

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # Start HTML content
    html = """
    <html>
    <head>
        <style>
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid black; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .positive { color: green; }
            .negative { color: red; }
            .chart { margin: 20px 0; }
        </style>
    </head>
    <body>
    """

    # Alerts section (from Binance scan)
    if alerts:
        html += """
        <h3>Alerts: current price > 250% of past 5-day low and funding rate > 0</h3>
        <table>
            <tr><th>Symbol</th><th>Price</th><th>5d Min</th><th>Price/Min</th><th>Funding Rate</th></tr>
        """
        for a in alerts:
            html += f"<tr><td>{a['symbol']}</td><td>{a['price']:.6f}</td><td>{a['min_low']:.6f}</td><td>{a['multiplier']:.2f}x</td><td>{a['funding_rate']:.8f}</td></tr>"
        html += "</table>"

    # Existing symbols table
    html += """
        <table>
            <tr>
                <th>Symbol</th>
                <th>Signal</th>
                <th>Last Price</th>
            </tr>
    """

    for symbol, data in symbols_data.items():
        signal = data['signal']
        price = data['price']
        signal_class = 'positive' if signal > 0 else 'negative'
        
        # Generate chart
        chart_base64 = create_signal_chart(data['market_data'], data['signals'], symbol)
        
        html += f"""
            <tr>
                <td>{symbol}</td>
                <td class="{signal_class}">{signal:.4f}</td>
                <td>{price:.4f}</td>
            </tr>
            <tr>
                <td colspan="3" class="chart">
                    <img src="data:image/png;base64,{chart_base64}" width="100%">
                </td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    msg.attach(MIMEText(html, 'html'))

    # If password not provided, skip sending email to avoid failing with credentials exposed
    if not sender_password:
        print("SENDER_PASSWORD not set; skipping email send.")
        return

    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")

def create_signal_chart(market_data, signals, symbol):
    """Create price and signal chart and return as base64 string"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
    
    # Plot price
    ax1.plot(market_data.index, market_data['close'], label='Price', color='blue')
    ax1.set_title(f'{symbol} Price')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # Plot signals
    ax2.plot(signals, label='Trading Signals', color='red')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.fill_between(range(len(signals)), 
                     signals,
                     0, 
                     where=(np.array(signals) > 0),
                     color='green', 
                     alpha=0.3,
                     label='Buy Signal')
    ax2.fill_between(range(len(signals)), 
                     signals,
                     0, 
                     where=(np.array(signals) < 0),
                     color='red', 
                     alpha=0.3,
                     label='Sell Signal')
    ax2.set_title('Trading Signals')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Signal Strength')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode()

def main(symbol='DOGE/USDT'):
    """Modified main function to return market data and signals"""
    # Load the trained model
    sequence_length = 48
    model = TorchNet(sequence_length)
    model_filename = f'models/best_model_{symbol.replace("/", "_")}.pth'
    
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"No trained model found for {symbol}. Please train the model first.")
    
    model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
    model.eval()

    # Get market data with both normalized and raw values
    data_loader = DataLoader(data_source='binance', symbol=symbol)
     # Get raw data
     
    normalized_market_data = data_loader.update_data()  # Get normalized data
    raw_market_data = data_loader.load_data(normalize=False) 
    # Use last 129 points for both datasets
    raw_market_data = raw_market_data.iloc[-100:-1]
    normalized_market_data = normalized_market_data.iloc[-100:-1]
    
    # Setup backtester
    backtester = Backtester(initial_capital=1000)
    

    # Prepare price data - use normalized data for model input but raw data for trading
    normalized_prices = normalized_market_data['close'].values
    raw_prices = raw_market_data['close'].values
    last_price = raw_prices[-1]
    last_signal = None
    
    # Store signals in a list
    signals = []
    for i in range(sequence_length, len(normalized_prices)):
        # Prepare input sequence using normalized data
        sequence = normalized_market_data[['volume', 'high', 'low', 'close']].values[i-sequence_length:i]
        sequence = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        
        # Get model prediction
        with torch.no_grad():
            signal = model(sequence).item()
            signal = np.clip(signal, -1, 1)  # Clip signal to [-1, 1]
            signals.append(signal)
            backtester.signal_history.append(signal)  # Record signal
            last_signal = signal

    return {
        'signal': last_signal, 
        'price': last_price,
        'market_data': raw_market_data.iloc[-len(signals):],  # Return raw market data
        'signals': signals
    }
def main2(symbol='LTC/USDT'):
    """Modified main function to return market data and signals"""
    # Load the trained model
    sequence_length = 32
    model = TorchNet(sequence_length,sequence_length)
    model_filename = f'models/best_model_{symbol.replace("/", "_")}1.pth'
    
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"No trained model found for {symbol}. Please train the model first.")
    
    model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
    model.eval()

    # Get market data with both normalized and raw values
    data_loader = DataLoader(data_source='binance', symbol=symbol)
     # Get raw data
     
    normalized_market_data = data_loader.load_data()  # Get normalized data
    raw_market_data = data_loader.load_data(normalize=False) 
    # Use last 129 points for both datasets
    raw_market_data = raw_market_data.iloc[-100:-1]
    normalized_market_data = normalized_market_data.iloc[-100:-1]
    
    # Setup backtester
    backtester = Backtester(initial_capital=1000)
    

    # Prepare price data - use normalized data for model input but raw data for trading
    normalized_prices = normalized_market_data['close'].values
    raw_prices = raw_market_data['close'].values
    last_price = raw_prices[-1]
    last_signal = None
    
    # Store signals in a list
    signals = []
    for i in range(sequence_length, len(normalized_prices)):
        # Prepare input sequence using normalized data
        sequence = normalized_market_data[['volume', 'high', 'low', 'close']].values[i-sequence_length:i]
        sequence = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        
        # Get model prediction
        with torch.no_grad():
            signal = model(sequence).item()
            signal = np.clip(signal, -1, 1)  # Clip signal to [-1, 1]
            signals.append(signal)
            backtester.signal_history.append(signal)  # Record signal
            last_signal = signal

    return {
        'signal': last_signal, 
        'price': last_price,
        'market_data': raw_market_data.iloc[-len(signals):],  # Return raw market data
        'signals': signals
    }

def main3(symbol='LINK/USDT'):
    """Modified main function to return market data and signals"""
    # Load the trained model
    sequence_length = 48
    model = TorchNet2(sequence_length)
    model_filename = f'models/best_model_{symbol.replace("/", "_")}3.pth'
    
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"No trained model found for {symbol}. Please train the model first.")
    
    model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
    model.eval()

    # Get market data with both normalized and raw values
    data_loader = DataLoader(data_source='binance', symbol=symbol)
     # Get raw data
     
    normalized_market_data = data_loader.load_data()  # Get normalized data
    raw_market_data = data_loader.load_data(normalize=False) 
    # Use last 129 points for both datasets
    raw_market_data = raw_market_data.iloc[-100:-1]
    normalized_market_data = normalized_market_data.iloc[-100:-1]
    
    # Setup backtester
    backtester = Backtester(initial_capital=1000)
    

    # Prepare price data - use normalized data for model input but raw data for trading
    normalized_prices = normalized_market_data['close'].values
    raw_prices = raw_market_data['close'].values
    last_price = raw_prices[-1]
    last_signal = None
    
    # Store signals in a list
    signals = []
    for i in range(sequence_length, len(normalized_prices)):
        # Prepare input sequence using normalized data
        sequence = normalized_market_data[['volume', 'high', 'low', 'close']].values[i-sequence_length:i]
        sequence = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        
        # Get model prediction
        with torch.no_grad():
            signal = model(sequence).item()
            signal = np.clip(signal, -1, 1)  # Clip signal to [-1, 1]
            signals.append(signal)
            backtester.signal_history.append(signal)  # Record signal
            last_signal = signal

    return {
        'signal': last_signal, 
        'price': last_price,
        'market_data': raw_market_data.iloc[-len(signals):],  # Return raw market data
        'signals': signals
    }
if __name__ == '__main__':
    symbols = [
    #     'CRV/USDT',
    #    'LTC/USDT',#1.17
    #    'LINK/USDT', #0.915
    #     'ETH/USDT',#1.07
       
     'LTC/USDT',#0.2670
          'LINK/USDT', #1.6
          'DOGE/USDT', 
    #      'ETH/USDT',#1.6311111111111
    #    'NEAR/USDT',
   #11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111      'DOGE/USDT',# 0.954
    #    'AAVE/USDT',#0.9017
    #  'NEAR/USDT',
    # #     'SOL/USDT',#0.6
    #     'AVAX/USDT',#0.58
    # #     'OP/USDT',#0.779
        
    ]

    symbols_data = {}
    for symbol in symbols:
        print(f"Processing {symbol}...")
        result = main(symbol)
        symbols_data[symbol] = result
    result2 = main2()
    symbols_data['LTC2/USDT'] = result2

    result3 = main3()
    symbols_data['LINK2/USDT'] = result3

    # scan Binance futures for alerting (can be slow for many symbols)
    # alerts = scan_binance_futures(min_multiplier=2.5, day_window=5, max_symbols=None)

    send_email(symbols_data)
