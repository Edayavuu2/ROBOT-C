import gym
import numpy as np

class PusherEnv(gym.Env):
    def __init__(self):
        super(PusherEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def step(self, action):
        # Ortama göre bir adım at ve yeni durumu döndür
        obs = np.random.rand(4)
        reward = -np.sum(np.square(action))  # Sahte ödül
        done = False
        return obs, reward, done, {}

    def reset(self):
        return np.random.rand(4)
