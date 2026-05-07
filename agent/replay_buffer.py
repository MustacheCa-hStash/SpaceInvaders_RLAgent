from collections import deque
import random

import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen = capacity)

    def append(self, state, action, reward, next_state, done):
        state = np.asarray(state, dtype = np.uint8)
        next_state = np.asarray(next_state, dtype = np.uint8)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(np.stack(states), dtype = torch.float32, device = device) / 255.0
        actions = torch.as_tensor(actions, dtype = torch.long, device = device)
        rewards = torch.as_tensor(rewards, dtype = torch.float32, device = device)
        next_states = torch.as_tensor(np.stack(next_states), dtype = torch.float32, device = device) / 255.0
        dones = torch.as_tensor(dones, dtype = torch.float32, device = device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)