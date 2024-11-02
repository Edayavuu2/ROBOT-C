import os
import torch
from sac import SAC  # SAC modelini içe aktar
from train import train  # Eğitim işlevini içe aktar
from models import PolicyNetwork, SoftQNetwork  # Modelleri içe aktar
from utility import ReplayBuffer, NormalizedActions  # Yardımcı dosyalar ve veri yapıları
import gym

# Ayarlar ve dizinler
project_directory = "path/to/your/project_directory"  # Proje dizininin yolunu belirt
models_directory = os.path.join(project_directory, "models")
os.makedirs(models_directory, exist_ok=True)  # Models dizini yoksa oluştur
replay_buffer = ReplayBuffer(int(1e6))  # Replay buffer ayarla

# Ortam ve SAC ajanını başlat
env = NormalizedActions(gym.make("Pusher-v2"))  # Pusher ortamını tanımla
agent = SAC(env, replay_buffer)

# Eğitimi başlat
train(agent)

# Eğitim sonrası modeli kaydet
torch.save(agent.policy_net.state_dict(), os.path.join(models_directory, "policy_net.pth"))
torch.save(agent.soft_q_net1.state_dict(), os.path.join(models_directory, "soft_q_net1.pth"))
torch.save(agent.soft_q_net2.state_dict(), os.path.join(models_directory, "soft_q_net2.pth"))

print("Model eğitimi tamamlandı ve modeller kaydedildi.")
