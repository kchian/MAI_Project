import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from numpy.random import randint

# Q-Value Network
class QNetwork(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=75):
        super().__init__()
        self.obs_size    = obs_size
        self.action_size = action_size
        self.epsilon     = 1
        self.net = nn.Sequential(nn.Linear(np.prod(obs_size), hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, action_size)) 
        
    def forward(self, obs):
        """
        Estimate q-values given obs

        Args:
            obs (tensor): current obs, size (batch x obs_size)

        Returns:
            q-values (tensor): estimated q-values, size (batch x action_size)
        """
        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1)
        return self.net(obs_flat)

'''
    def get_action(self, obs, q_network, epsilon, allow_break_action):
        """
        Select action according to e-greedy policy

        Args:
            obs (np-array): current observation, size (obs_size)
            q_network (QNetwork): Q-Network
            epsilon (float): probability of choosing a random action

        Returns:
            action (int): chosen action [0, action_size)
        """
        # Prevent computation graph from being calculated
        with torch.no_grad():
            # Calculate Q-values fot each action
            obs_torch = torch.tensor(obs.copy(), dtype=torch.float).unsqueeze(0)
            action_values = self(obs_torch)

            # Remove attack/mine from possible actions if not facing a diamond
            if not allow_break_action:
                action_values[0, 3] = -float('inf')  

            #Get random action if epsilon else Q-value
            if randint(1000)/1000 < epsilon:
                action_idx = randint(4)
                if not allow_break_action:
                    action_idx = randint(3)
            else:
            # Select action with highest Q-value
                action_idx = torch.argmax(action_values).item()
            
        return action_idx

    def prepare_batch(self, replay_buffer):
        """
        Randomly sample batch from replay buffer and prepare tensors

        Args:
            replay_buffer (list): obs, action, next_obs, reward, done tuples

        Returns:
            obs (tensor): float tensor of size (BATCH_SIZE x obs_size
            action (tensor): long tensor of size (BATCH_SIZE)
            next_obs (tensor): float tensor of size (BATCH_SIZE x obs_size)
            reward (tensor): float tensor of size (BATCH_SIZE)
            done (tensor): float tensor of size (BATCH_SIZE)
        """
        batch_data = random.sample(replay_buffer, BATCH_SIZE)
        obs = torch.tensor([x[0] for x in batch_data], dtype=torch.float)
        action = torch.tensor([x[1] for x in batch_data], dtype=torch.long)
        next_obs = torch.tensor([x[2] for x in batch_data], dtype=torch.float)
        reward = torch.tensor([x[3] for x in batch_data], dtype=torch.float)
        done = torch.tensor([x[4] for x in batch_data], dtype=torch.float)
        
        return obs, action, next_obs, reward, done
    
    def learn(self, batch, q_network, target_network):
        """
        Update Q-Network according to DQN Loss function

        Args:
            batch (tuple): tuple of obs, action, next_obs, reward, and done tensors
            optim (Adam): Q-Network optimizer
            q_network (QNetwork): Q-Network
            target_network (QNetwork): Target Q-Network
        """
        obs, action, next_obs, reward, done = batch

        self.optimizer.zero_grad()
        values = self(obs).gather(1, action.unsqueeze(-1)).squeeze(-1)
        target = torch.max(target_network(next_obs), 1)[0]
        target = reward + GAMMA * target * (1 - done)
        loss = torch.mean((target - values) ** 2)
        loss.backward()
        self.optimizer.step()

        return loss.item()
'''