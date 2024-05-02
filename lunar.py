import gymnasium
import gym.wrappers
import numpy as np
from PIL import Image
import torch


class DQN_Lunar(gym.Wrapper): 
    
    def __init__(self, render_mode='rgb_array', device='cpu'):
        env = gymnasium.make("LunarLander-v2", render_mode=render_mode)
        super(DQN_Lunar, self).__init__(env)
        self.device = device
        self.actions = env.action_space.n
        self.in_dim = np.array(env.observation_space.shape).prod()


    def step(self, action):
        next_state, reward, terminated, truncated, infos  = self.env.step(action)
        return next_state, reward, terminated, truncated, infos 
    
    def reset(self):
        return self.env.reset()