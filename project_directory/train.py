
from pusher_env import PusherEnv
import numpy as np
from project_directory.sac import SAC
from project_directory.utility import ReplayBuffer


env = PusherEnv()
sac = SAC(state_dim=4, action_dim=2, max_action=1)
replay_buffer = ReplayBuffer()

# Eğitim döngüsü
for episode in range(1000):
    state = env.reset()
    for t in range(200):
        action = sac.select_action(np.array(state))
        next_state, reward, done, _ = env.step(action)

        replay_buffer.add((state, next_state, action, reward, done))

        state = next_state
        if done:
            break

    if episode % 100 == 0:
        print(f"Episode {episode} completed")
