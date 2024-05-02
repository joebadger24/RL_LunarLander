import numpy as np
import gymnasium
import torch
import random
from lunar import *
from network import *
from agent import *
import os



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

environment = DQN_Lunar(device=device, render_mode='human')
out_dim = environment.actions
in_dim = environment.in_dim
model = Network(in_dim=in_dim, out_dim=out_dim)
model.to(device)

# Define which model state to test:

#model.load(start_new=False, filename="network/latest.pt")
model.load(start_new=False, filename="models/model_iter_1000.pt")


agent = PER_Agent(
            model=model, device=device, nb_actions=out_dim,
            epsilon= 0, min_epsilon=0, epsilon_decay=0, 
            gamma=0.99, learning_rate=0.001, 
            capacity=2000000, batch_size=64,
            max_step_ep=2000, update_frequency=1000,
            env_dims=[environment.observation_space, environment.action_space],
            alpha = 0.2, beta = 0.5, beta_annealing = 0.000006, duelling=False
        )

import time
time.sleep(2)

agent.test(env=environment)
