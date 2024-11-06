from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.buffer = deque(maxlen=int(max_size))
        self.state_dim = state_dim
        self.action_dim = action_dim

    def add(self, state, action, reward, next_state, done):
        # Verilerin doğru boyutta olup olmadığını kontrol edin
        if len(state) != self.state_dim or len(action) != self.action_dim:
            raise ValueError("State or action dimension does not match the specified dimensions.")
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples in the buffer to sample.")
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        not_dones = 1 - dones  # `done` değerini `not_done` şeklinde dönüştür
        return states, actions, rewards, next_states, not_dones

    def __len__(self):
        return len(self.buffer)  # Buffer'daki mevcut örnek sayısını döndür
