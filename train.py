from sac import SAC
from pusher_env import PusherEnv

env = PusherEnv()
sac = SAC(env)
sac.train(1000)  # 1000 bölüm eğitimi
