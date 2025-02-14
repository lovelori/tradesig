import torch
import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import DataLoader
from models.torch_net import TorchNet
from data.dataset import CryptoDataset
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import base64
from io import BytesIO
import matplotlib
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
def send_email(symbols_data):
    """Modified email function to include charts"""
    smtp_server = "smtp.163.com"
    smtp_port = 465
    sender_email = "13972206966@163.com"
    sender_password = "GFDQbKgycdyNC2pT"
    receiver_email = "cdha0@outlook.com"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f"Trading Signals Update - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # Create HTML table
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
        <h2>Latest Trading Signals</h2>
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
    model = TorchNet()
    model_filename = f'models/{symbol.replace("/", "_")}_model.pth'
    
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"No trained model found for {symbol}. Please train the model first.")
    
    model.load_state_dict(torch.load(model_filename))
    model.eval()

    # Get market data
    data_loader = DataLoader(data_source='binance', symbol=symbol)
    market_data = data_loader.update_data() #limit=15000,use_cache=True, normalize=True,write_cache=True
    # Only use the most recent 50% of data
    #half_point = int(len(market_data) *0.7)
    market_data = market_data.iloc[-129:]
    
    # Setup backtester
    backtester = Backtester(initial_capital=1000)
    sequence_length = 99

    # Prepare price data
    prices = market_data['close'].values
    last_price = prices[-1]
    last_signal = None
    
    # Store signals in a list
    signals = []
    for i in range(sequence_length, len(prices)):
        # Prepare input sequence
        sequence = market_data[['open', 'high', 'low', 'close', 'volume']].values[i-sequence_length:i]
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
        'market_data': market_data.iloc[-30:],
        'signals': signals
    }

if __name__ == '__main__':
    symbols = [
        'ETH/USDT',
        'LTC/USDT',
        'LINK/USDT',
        'DOGE/USDT',
        'AAVE/USDT',
        'GRT/USDT',
        '1INCH/USDT',
    ]

    symbols_data = {}
    for symbol in symbols:
        print(f"Processing {symbol}...")
        result = main(symbol)
        symbols_data[symbol] = result
        
    send_email(symbols_data)