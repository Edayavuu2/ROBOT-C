import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal  # Doğrudan Normal dağılımını import et

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        # Actor ağı
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )

        # Kritik ağı
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Standart sapma parametresi
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def act(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        
        # Ortalama ve standart sapma
        mu = self.pi(obs)
        std = torch.exp(self.log_std)
        
        # Normal dağılımı kullanarak eylem dağılımı oluştur
        action_dist = Normal(mu, std)
        action = action_dist.rsample()  # Eylemi rsample ile al (gradient takip edilebilir)
        log_prob = action_dist.log_prob(action).sum(axis=-1)  # Log-olasılık hesapla
        return action, log_prob

    def q(self, obs, act):
        # Q ağı üzerinden Q değerini hesapla
        return self.q_network(torch.cat([obs, act], dim=-1))
