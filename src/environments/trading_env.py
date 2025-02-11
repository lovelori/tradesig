import gymnasium as gym
import numpy as np
from gymnasium import spaces
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        logger.info(f"Initialized environment with {self.max_steps + 1} data points")
        
        # Define action space (0: hold, 1: buy, 2: sell)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space (price data + account info)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6,),  # OHLCV + position
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.done = False
        
        logger.debug("Environment reset")
        return self._get_observation(), {}
    
    def step(self, action):
        if self.done:
            logger.warning("Step called on done environment")
            return self._get_observation(), 0, True, False, {}
            
        current_price = float(self.data.iloc[self.current_step]['close'])
        
        # Execute trading action
        reward = 0
        if action == 1 and self.position == 0:  # Buy
            self.position = self.balance / current_price
            self.balance = 0
            logger.debug(f"Buy at price: {current_price}")
        elif action == 2 and self.position > 0:  # Sell
            self.balance = self.position * current_price
            self.position = 0
            logger.debug(f"Sell at price: {current_price}")
                
        # Move to next step
        self.current_step += 1
        self.done = self.current_step >= self.max_steps
        
        reward = self._calculate_reward()
        observation = self._get_observation()
        
        return observation, reward, self.done, False, {}
    
    def _get_observation(self):
        if self.current_step > self.max_steps:
            logger.warning("Attempting to get observation beyond data range")
            return np.zeros(6, dtype=np.float32)
            
        try:
            current_data = self.data.iloc[self.current_step]
            obs = np.array([
                float(current_data['open']),
                float(current_data['high']),
                float(current_data['low']),
                float(current_data['close']),
                float(current_data['volume']),
                float(self.position)
            ], dtype=np.float32)
            return obs
        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            return np.zeros(6, dtype=np.float32)
    
    def _calculate_reward(self):
        if self.current_step == 0:
            return 0.0
            
        try:
            current_value = self.balance
            if self.position > 0:
                current_value = self.position * float(self.data.iloc[self.current_step]['close'])
                
            prev_value = self.balance
            if self.position > 0:
                prev_value = self.position * float(self.data.iloc[self.current_step-1]['close'])
                
            return float((current_value - prev_value) / prev_value)
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0