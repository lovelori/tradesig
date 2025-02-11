from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TradingAgent:
    def __init__(self, env):
        """
        Initialize the trading agent
        
        Args:
            env: Gymnasium environment for trading
        """
        try:
            self.env = env
            self.model = PPO("MlpPolicy", env, verbose=1)
        except Exception as e:
            print(f"Error initializing agent: {e}")
            raise

    def train(self, total_timesteps):
        """
        Train the agent
        
        Args:
            total_timesteps (int): Number of timesteps to train for
        """
        try:
            self.model.learn(total_timesteps=total_timesteps)
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def predict(self, obs):
        """
        Make a prediction for given observation
        
        Args:
            obs: Environment observation
        Returns:
            tuple: Action and state
        """
        try:
            return self.model.predict(obs)
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None

    def save_model(self, path):
        """
        Save the model to disk
        
        Args:
            path (str): Path to save the model
        """
        try:
            self.model.save(path)
        except Exception as e:
            print(f"Error saving model: {e}")
            raise

    def load_model(self, path):
        """
        Load a model from disk
        
        Args:
            path (str): Path to the saved model
        """
        try:
            self.model = PPO.load(path, env=self.env)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise