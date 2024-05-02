import numpy as np
import gymnasium
import torch
import random
from lunar import *
from network import *
from agent import *
import os
import matplotlib.pyplot as plt

data_ed_001 = []
data_ed_005 = []
data_ed_010 = []
data_ed_020 = []


def param_PER_dqn(num_agents, num_steps, data_list, epsilon):
    for i in range(num_agents):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        environment = DQN_Lunar(device=device, render_mode='rgb_array')
        out_dim = environment.actions
        in_dim = environment.in_dim
        model = Network(in_dim=in_dim, out_dim=out_dim)
        model.to(device)
        model.load(start_new=True)

        delete_existing_files = True

        if delete_existing_files:
            files = os.listdir('models')
            for file in files:
                os.remove(os.path.join('models', file))
            files = os.listdir('network')
            for file in files:
                os.remove(os.path.join('network', file))
            print("Deleted Exisiting Files")
                    

        agent = PER_Agent(
            model=model, device=device, nb_actions=out_dim,
            epsilon= 1.0, min_epsilon=0.1, epsilon_decay=epsilon, 
            gamma=0.99, learning_rate=0.001, 
            capacity=2000000, batch_size=64,
            max_step_ep=2000, update_frequency=1000,
            env_dims=[environment.observation_space, environment.action_space],
            alpha = 0.2, beta = 0.5, beta_annealing = 0.000006, duelling=False
        )

        data_set = agent.train(env=environment, total_steps=num_steps)
        data_list.append(data_set['Returns'])
        print(f"Agent{i+1}/{num_agents} complete")


def process_data(data):
    if data == []:
        return []
    else:
        min_episodes = min(len(agent) for agent in data)
        padded_returns = [agent[:min_episodes] + [0] * (min_episodes - len(agent)) for agent in data]
        return [sum(ep[i] for ep in padded_returns) / len(data) for i in range(min_episodes)]


# To load data from txt files, make this True
# To run the model again, make this False

load_data = True



if load_data:
    data_ed_001_returns = np.loadtxt('data/data_ed_001_returns.txt')
    data_ed_005_returns = np.loadtxt('data/data_ed_005_returns.txt')
    data_ed_010_returns = np.loadtxt('data/data_ed_010_returns.txt')
    data_ed_020_returns = np.loadtxt('data/data_ed_020_returns.txt')
else:
    param_PER_dqn(1, 450_000, data_ed_001, 0.001)
    param_PER_dqn(1, 450_000, data_ed_005, 0.005)
    param_PER_dqn(1, 450_000, data_ed_010, 0.010)
    param_PER_dqn(1, 450_000, data_ed_020, 0.020)
    data_ed_001_returns = process_data(data_ed_001)
    data_ed_005_returns = process_data(data_ed_005)
    data_ed_010_returns = process_data(data_ed_010)
    data_ed_020_returns = process_data(data_ed_020)


np.savetxt('data/data_ed_001_returns.txt', data_ed_001_returns)
np.savetxt('data/data_ed_005_returns.txt', data_ed_005_returns)
np.savetxt('data/data_ed_010_returns.txt', data_ed_010_returns)
np.savetxt('data/data_ed_020_returns.txt', data_ed_020_returns)


max_x_value = min(len(data_ed_001_returns), len(data_ed_005_returns), len(data_ed_010_returns), len(data_ed_020_returns))

def moving_average(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        if i < window_size - 1:
            smoothed_data.append(sum(data[:i+1]) / (i + 1))
        else:
            smoothed_data.append(sum(data[i-window_size+1:i+1]) / window_size)
    return smoothed_data

window = 100

data_ed_001_returns = moving_average(data_ed_001_returns, window)
data_ed_005_returns = moving_average(data_ed_005_returns, window)
data_ed_010_returns = moving_average(data_ed_010_returns, window)
data_ed_020_returns = moving_average(data_ed_020_returns, window)

plt.plot(data_ed_001_returns, label='ed=0.001')
plt.plot(data_ed_005_returns, label='ed=0.005')
plt.plot(data_ed_010_returns, label='ed=0.010')
plt.plot(data_ed_020_returns, label='ed=0.020')
plt.axhline(y=200, color='b', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Plot of Rewards')
plt.legend()
plt.xlim(left=0, right=max_x_value)
plt.show()
