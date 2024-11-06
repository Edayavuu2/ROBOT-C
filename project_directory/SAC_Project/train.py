import gymnasium as gym
from sac import SACAgent
from replay_buffer import ReplayBuffer
import numpy as np

# Ortam ve ajan kurulumu
env = gym.make('Pusher-v4')
agent = SACAgent(env)

# Replay buffer'ı başlat ve agent'a ata
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
replay_buffer = ReplayBuffer(max_size=100000, state_dim=state_dim, action_dim=action_dim)
agent.set_replay_buffer(replay_buffer)  # replay buffer'ı ajan ile ilişkilendir

num_episodes = 500  # Eğitim süresi
save_interval = 100  # Modelin kaydedilme sıklığı
batch_size = 64  # Güncelleme için batch boyutu

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    print(f"Starting Episode {episode + 1}/{num_episodes}")  # Eğitim başlangıcında bilgi ver

    while not done:
        action = agent.act(obs)
        next_obs, reward, done, _, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, done)  # Replay buffer'a ekle
        obs = next_obs
        total_reward += reward

        # Replay buffer'da yeterli örnek varsa güncelleme yap
        if len(replay_buffer) >= batch_size:
            agent.update(batch_size=batch_size)  # Güncelleme yap

    # Episod sonu çıktıları
    print(f"Episode {episode + 1} - Total Reward: {total_reward}")
    with open("training_log.txt", "a") as f:
        f.write(f"Episode {episode + 1} - Total Reward: {total_reward}\n")

    # Belirli aralıklarla model kaydetme
    if episode % save_interval == 0:
        agent.save(f'checkpoints/sac_model_{episode}.pt')
        print(f"Model saved at Episode {episode + 1}")  # Model kaydedildiğinde bilgi ver
