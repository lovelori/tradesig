from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Define network dimensions
        self.features_dim = 128
        
        # Custom network architecture
        self.shared_net = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, self.features_dim),
            nn.ReLU()
        )
        
        # Policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(self.features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space.n)
        )
        
        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(self.features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, deterministic=False):
        """
        Forward pass in all the networks
        """
        # Convert observation to tensor and move to correct device
        obs = th.as_tensor(obs).float().to(self.device)
        
        # Get features and ensure they're on the correct device
        shared_features = self.shared_net(obs)
        action_logits = self.policy_net(shared_features)
        values = self.value_net(shared_features)
        
        # Create action distribution
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        
        return distribution, values, shared_features

    def _predict(self, observation, deterministic=False):
        """
        Get the action according to the policy for a given observation.

        Args:
            observation: Observation to use for prediction
            deterministic: Whether to use deterministic or stochastic actions

        Returns:
            Tuple: (actions, values, log_probs)
        """
        with th.no_grad():
            observation = th.as_tensor(observation).float().to(self.device)
            distribution, values, _ = self.forward(observation)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            
        return actions, values, log_prob

    def forward_actor(self, obs, deterministic=False):
        shared_features = self.shared_net(obs)
        return self.policy_net(shared_features)

    def forward_critic(self, obs):
        shared_features = self.shared_net(obs)
        return self.value_net(shared_features)

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy
        """
        # Ensure inputs are on the correct device
        obs = th.as_tensor(obs).float().to(self.device)
        actions = th.as_tensor(actions).long().to(self.device)
        
        distribution, values, _ = self.forward(obs)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy

class TradingAgent:
    def __init__(self, env, algorithm="DQN"):
        try:
            self.env = env
            device = "cuda" if th.cuda.is_available() else "cpu"
            
            if algorithm == "PPO":
                self.model = PPO("MlpPolicy", env, verbose=1)
            elif algorithm == "DQN":
                self.model = DQN("MlpPolicy", env, verbose=1)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
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