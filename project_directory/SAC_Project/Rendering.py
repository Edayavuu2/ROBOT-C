"""
# Görselleştirme için eğitimli modeli kullanma

import gymnasium as gym
from sac import SACAgent
import torch

env = gym.make('Pusher-v4')
agent = SACAgent(env)
agent.load('checkpoints/sac_model_final.pt')  # Eğitilmiş modelin yolunu girin

obs, _ = env.reset()
done = False

while not done:
    action = agent.act(obs)
    obs, _, done, _, _ = env.step(action)
    env.render()  # Ortamı görselleştir
    """