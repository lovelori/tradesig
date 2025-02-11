def log_message(message):
    """Logs a message to the console."""
    print(f"[LOG] {message}")

def calculate_reward(profit, transaction_cost):
    """Calculates the reward based on profit and transaction cost."""
    return profit - transaction_cost

def preprocess_data(data):
    """Preprocesses the input data for model training."""
    # Implement preprocessing logic here
    return data

def format_timestamp(timestamp):
    """Formats a timestamp into a readable string."""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")