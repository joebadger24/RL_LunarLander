import numpy as np
import gymnasium
import torch
import random
from lunar import *
from network import *
from agent import *
import os
import matplotlib.pyplot as plt

dqn_data = []
duelling_dqn_data = []
PER_dqn_data = []
PER_duelling_dqn_data = []
random_agent_data = []
num_agents = 1
num_steps = 600_000


def train_dqn(num_agents, num_steps):
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
                    

        agent = no_PER_Agent(
            model=model, device=device, nb_actions=out_dim,
            epsilon= 1.0, min_epsilon=0.1, epsilon_decay=0.010, 
            gamma=0.99, learning_rate=0.001, 
            capacity=2000000, batch_size=64,
            max_step_ep=2000, update_frequency=1000,
            env_dims=[environment.observation_space, environment.action_space],
            alpha = 0.2, beta = 0.5, beta_annealing = 0.000006, duelling=False
        )

        data_set = agent.train(env=environment, total_steps=num_steps)
        dqn_data.append(data_set['Returns'])
        print(f"Agent{i+1}/{num_agents} complete")

def train_duelling_dqn(num_agents, num_steps):
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
                    

        agent = no_PER_Agent(
            model=model, device=device, nb_actions=out_dim,
            epsilon= 1.0, min_epsilon=0.1, epsilon_decay=0.010, 
            gamma=0.99, learning_rate=0.001, 
            capacity=2000000, batch_size=64,
            max_step_ep=2000, update_frequency=1000,
            env_dims=[environment.observation_space, environment.action_space],
            alpha = 0.2, beta = 0.5, beta_annealing = 0.000006, duelling=True
        )

        data_set = agent.train(env=environment, total_steps=num_steps)
        duelling_dqn_data.append(data_set['Returns'])
        print(f"Agent{i+1}/{num_agents} complete")

def train_PER_dqn(num_agents, num_steps):
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
            epsilon= 1.0, min_epsilon=0.1, epsilon_decay=0.010, 
            gamma=0.99, learning_rate=0.001, 
            capacity=2000000, batch_size=64,
            max_step_ep=2000, update_frequency=1000,
            env_dims=[environment.observation_space, environment.action_space],
            alpha = 0.2, beta = 0.5, beta_annealing = 0.000006, duelling=False
        )

        data_set = agent.train(env=environment, total_steps=num_steps)
        PER_dqn_data.append(data_set['Returns'])
        print(f"Agent{i+1}/{num_agents} complete")
    
def train_PER_duelling_dqn(num_agents, num_steps):
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
            epsilon= 1.0, min_epsilon=0.1, epsilon_decay=0.010, 
            gamma=0.99, learning_rate=0.001, 
            capacity=2000000, batch_size=64,
            max_step_ep=2000, update_frequency=1000,
            env_dims=[environment.observation_space, environment.action_space],
            alpha = 0.2, beta = 0.5, beta_annealing = 0.000006, duelling=True
        )

        data_set = agent.train(env=environment, total_steps=num_steps)
        PER_duelling_dqn_data.append(data_set['Returns'])
        print(f"Agent{i+1}/{num_agents} complete")

def train_random_agent(num_agents, num_steps):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    environment = DQN_Lunar(device=device, render_mode='rgb_array')
    out_dim = environment.actions
    in_dim = environment.in_dim
    model = Network(in_dim=in_dim, out_dim=out_dim)
    model.to(device)
    model.load(start_new=True)

    agent = random_Agent(
            model=model, device=device, nb_actions=out_dim,
            epsilon= 1.0, min_epsilon=0.1, epsilon_decay=0.005, 
            gamma=0.99, learning_rate=0.001, 
            capacity=2000000, batch_size=64,
            max_step_ep=2000, update_frequency=1000,
            env_dims=[environment.observation_space, environment.action_space],
            alpha = 0.2, beta = 0.5, beta_annealing = 0.000006, duelling=True
        )

    data_set = agent.train(env=environment, total_steps=num_steps)
    random_agent_data.append(data_set['Returns'])
    print(f"Agent complete")



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
    dqn_average_returns = np.loadtxt('data/dqn_average_returns.txt')
    duelling_dqn_average_returns = np.loadtxt('data/duelling_dqn_average_returns.txt')
    PER_dqn_average_returns = np.loadtxt('data/PER_dqn_average_returns.txt')
    PER_duelling_dqn_average_returns = np.loadtxt('data/PER_duelling_dqn_average_returns.txt')
    random_agent_returns = np.loadtxt('data/random_agent_returns.txt')
else:
    train_dqn(num_agents, num_steps)
    train_duelling_dqn(num_agents, num_steps)
    train_PER_dqn(num_agents, num_steps)
    train_PER_duelling_dqn(num_agents, num_steps)
    train_random_agent(num_agents, num_steps)
    dqn_average_returns = process_data(dqn_data)
    duelling_dqn_average_returns = process_data(duelling_dqn_data)
    PER_dqn_average_returns = process_data(PER_dqn_data)
    PER_duelling_dqn_average_returns = process_data(PER_duelling_dqn_data)
    random_agent_returns = process_data(random_agent_data)

np.savetxt('data/dqn_average_returns.txt', dqn_average_returns)
np.savetxt('data/duelling_dqn_average_returns.txt', duelling_dqn_average_returns)
np.savetxt('data/PER_dqn_average_returns.txt', PER_dqn_average_returns)
np.savetxt('data/PER_duelling_dqn_average_returns.txt', PER_duelling_dqn_average_returns)
np.savetxt('data/random_agent_returns.txt',random_agent_returns)

max_x_value = min(len(dqn_average_returns), len(duelling_dqn_average_returns), len(PER_dqn_average_returns), len(PER_duelling_dqn_average_returns))

def moving_average(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        if i < window_size - 1:
            smoothed_data.append(sum(data[:i+1]) / (i + 1))
        else:
            smoothed_data.append(sum(data[i-window_size+1:i+1]) / window_size)
    return smoothed_data

window = 100

dqn_average_returns = moving_average(dqn_average_returns, window)
duelling_dqn_average_returns = moving_average(duelling_dqn_average_returns, window)
PER_dqn_average_returns = moving_average(PER_dqn_average_returns, window)
PER_duelling_dqn_average_returns = moving_average(PER_duelling_dqn_average_returns, window)
random_agent_returns = moving_average(random_agent_returns, window)


plt.plot(dqn_average_returns, label='DQN')
plt.plot(duelling_dqn_average_returns, label='Duelling DQN')
plt.plot(PER_dqn_average_returns, label='PER DQN')
plt.plot(PER_duelling_dqn_average_returns, label='PER Duelling DQN')
plt.plot(random_agent_returns, label='Random')
plt.axhline(y=200, color='b', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Plot of Rewards')
plt.legend()
plt.xlim(left=0, right=max_x_value)
plt.show()
