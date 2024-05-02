import random
import torch
import copy
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from collections import namedtuple


class PrioritizedReplayBuffer:
    def __init__(self, capacity, obs_shape, action_shape, alpha=0.6, beta=0.4, beta_annealing=1e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.pos = 0
        self.size = 0
        self.obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((capacity,) + action_shape, dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0
        self.min_priority = 1e-5
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.experience = namedtuple("Experience", field_names=["obs", "action", "reward", "next_obs", "done"])

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = done
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha / np.sum(priorities ** self.alpha)
        indices = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)
        beta = min(1, self.beta + self.beta_annealing)
        obs_batch = self.obs[indices]
        action_batch = self.actions[indices]
        reward_batch = self.rewards[indices]
        next_obs_batch = self.next_obs[indices]
        done_batch = self.dones[indices]
        return (
            obs_batch,
            action_batch,
            reward_batch,
            next_obs_batch,
            done_batch,
            weights,
            indices
        )

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = np.maximum(priorities, self.min_priority)
        self.max_priority = max(self.max_priority, np.max(self.priorities))

class PER_Agent:

    def __init__(self, model, device='cpu', duelling = True,  epsilon=1.0, min_epsilon=0.1, nb_actions=4, 
                 capacity=10000, batch_size = 32, learning_rate=0.00025, epsilon_decay = 0.05, 
                 gamma =0.99, max_step_ep = 2000, update_frequency = 1000, env_dims=None,
                 alpha = 0.2, beta = 0.5, beta_annealing = 0.000006):

        self.memory = PrioritizedReplayBuffer(
            capacity = capacity, obs_shape=(8,), action_shape=nb_actions, 
            alpha=alpha, beta=beta, beta_annealing=beta_annealing)

        self.model = model
        self.target_model = copy.deepcopy(model).eval()
        self.duelling = duelling
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = gamma
        self.nb_actions = nb_actions
        self.learning_rate = learning_rate
        self.max_step_ep = max_step_ep
        self.update_frequency = update_frequency
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.capacity = capacity

        self.optimiser = optim.Adam(model.parameters(), lr=learning_rate)


    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(0, self.nb_actions, (1,)).item()
        else:
            action_int = self.model(state)
            torch_int = torch.argmax(action_int, dim=0, keepdim=True)
            return torch_int.item()
        
    def compute_td_error(self, states, actions, rewards, next_states, dones):

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards,  dtype=torch.float32)
        dones = torch.tensor(dones,  dtype=torch.float32)

        q_values = self.model(states).gather(1, actions.unsqueeze(1))

        if self.duelling:
            with torch.no_grad():
                next_qsa_actions = self.model(next_states).argmax(dim=1)
                next_q_values = self.target_model(next_states).detach().gather(1, next_qsa_actions.unsqueeze(1))
                target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
                td_errors = F.smooth_l1_loss(target_q_values.detach(), q_values).detach()
        else:
            with torch.no_grad():
                next_q_values = self.target_model(next_states).max(dim=1, keepdim=True)[0]
                target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
                td_errors = F.smooth_l1_loss(target_q_values.detach(), q_values).detach()

        return td_errors.cpu().numpy()


    def train(self, env, total_steps):
        stats = {'Returns' : [], 'AvgReturns': [], 'EpsilonCheckPoint': []}
        ep_reward = 0
        state, _ = env.reset()
        episode_count = 0
        

        for step in range(1, total_steps + 1):
            
            action = self.get_action(state)
            next_state, reward, terminated, truncated, infos  = env.step(action)
            terminal = terminated or truncated
            ep_reward += reward     

            self.memory.add(
                state.reshape(1, -1), 
                np.array(action)[np.newaxis, ...], 
                np.array(reward)[np.newaxis, ...], 
                next_state.reshape(1, -1), 
                np.array(terminal)[np.newaxis, ...])

            state = next_state

            # Training Logic
            if step >= self.batch_size and step % 10 == 0:

                state_b, action_b, reward_b, next_state_b, terminal_b, weights_b, indices_b = self.memory.sample(self.batch_size)

                td_errors = self.compute_td_error(state_b, action_b, reward_b, next_state_b, terminal_b)
                self.memory.update_priorities(indices_b, td_errors)
                weights = torch.tensor(weights_b, dtype=torch.float32)

                state_b = torch.tensor(state_b, dtype=torch.float32)
                next_state_b = torch.tensor(next_state_b, dtype=torch.float32)
                action_b = torch.tensor(action_b, dtype=torch.long)
                reward_b = torch.tensor(reward_b,  dtype=torch.float32)
                terminal_b = torch.tensor(terminal_b,  dtype=torch.float32)

                 
                qsa_b = self.model(state_b).gather(1, action_b.unsqueeze(1))

                if self.duelling:
                    with torch.no_grad():
                        next_qsa_actions = self.model(next_state_b).argmax(dim=1)
                        next_q_values = self.target_model(next_state_b).detach().gather(1, next_qsa_actions.unsqueeze(1))
                        target_q_values = reward_b.unsqueeze(1) + (1 - terminal_b.unsqueeze(1)) * self.gamma * next_q_values

                else:
                    with torch.no_grad():
                        next_q_values = self.target_model(next_state_b).max(dim=1, keepdim=True)[0]
                        target_q_values = reward_b.unsqueeze(1) + (1 - terminal_b.unsqueeze(1)) * self.gamma * next_q_values


                loss = (weights * F.smooth_l1_loss(qsa_b, target_q_values.detach())).mean()

                self.optimiser.zero_grad()     # Zero the gradients
                loss.backward()                # Backpropagation
                self.optimiser.step()          # Update the neural network parameters

            if step % self.update_frequency == 0:
                for target_network_param, q_network_param in zip(self.target_model.parameters(), self.model.parameters()):
                    target_network_param.data.copy_(
                        1.0 * q_network_param.data + (1.0 - 1.0) * target_network_param.data
                    )
                    

            if terminal:
                stats['Returns'].append(ep_reward)
                ep_reward = 0
                state, _ = env.reset()
                episode_count += 1

                # Epsilon Decay
                if self.epsilon > self.min_epsilon:
                    self.epsilon = self.epsilon - self.epsilon_decay

                
                # Save copies of training stages
                if episode_count == 1:
                    self.model.save(filename=f"models/model_iter_{episode_count}.pt")

                if episode_count % 100 == 0:
                    print(f"Epsiode {episode_count} Average Returns {np.mean(stats['Returns'][-10:])} Epsilon {self.epsilon}")
                    self.model.save(filename=f"models/model_iter_{episode_count}.pt")
          
                
        return stats
    

    def test(self, env):
        for episode in range(5):
            total_reward = 0
            state, _ = env.reset()
            terminal = False
            for _ in range(self.max_step_ep):
                action = self.get_action(state)
                state, reward, terminal, _, _ = env.step(action)
                total_reward += reward
                if terminal:
                    break
            print(total_reward)

class no_PER_Agent:

    def __init__(self, model, device='cpu', duelling = True, epsilon=1.0, min_epsilon=0.1, nb_actions=4, 
                 capacity=10000, batch_size = 32, learning_rate=0.00025, epsilon_decay = 0.05, 
                 gamma =0.99, max_step_ep = 2000, update_frequency = 1000, env_dims=None,
                 alpha = 0.2, beta = 0.5, beta_annealing = 0.000006):
        
        self.rb = ReplayBuffer(
            capacity,
            env_dims[0],
            env_dims[1],
            device,
            handle_timeout_termination=False,
        )
        self.model = model
        self.target_model = copy.deepcopy(model).eval()
        self.duelling = duelling
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = gamma
        self.nb_actions = nb_actions
        self.learning_rate = learning_rate
        self.max_step_ep = max_step_ep
        self.update_frequency = update_frequency
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.capacity = capacity

        self.optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(0, self.nb_actions, (1,)).item()
        else:
            action_int = self.model(state)
            torch_int = torch.argmax(action_int, dim=0, keepdim=True)
            return torch_int.item()


    def train(self, env, total_steps):
        stats = {'Returns' : [], 'AvgReturns': [], 'EpsilonCheckPoint': []}
        ep_reward = 0
        state, _ = env.reset()
        episode_count = 0
        

        for step in range(1, total_steps + 1):
            
            action = self.get_action(state)
            next_state, reward, terminated, truncated, infos  = env.step(action)
            terminal = terminated or truncated
            ep_reward += reward     


            self.rb.add(state.reshape(1, -1), 
                        next_state.reshape(1, -1), 
                        np.array(action)[np.newaxis, ...], 
                        np.array(reward)[np.newaxis, ...], 
                        np.array(terminal)[np.newaxis, ...], 
                        np.array(0)[np.newaxis, ...],
                        )
            
            state = next_state

            # Training Logic
            if step >= self.batch_size and step % 10 == 0:

                data = self.rb.sample(self.batch_size)
                state_b =  data.observations
                action_b = data.actions
                reward_b = data.rewards
                terminal_b = data.dones
                next_state_b = data.next_observations

                qsa_b = self.model(state_b).gather(1, action_b).squeeze()

                
                if self.duelling:
                    with torch.no_grad():
                        next_qsa_actions = self.model(next_state_b).argmax(dim=1)
                        next_q_values = self.target_model(next_state_b).detach().gather(1, next_qsa_actions.unsqueeze(1)).squeeze()
                        target_q_values = reward_b.flatten() + (1 - terminal_b.flatten()) * self.gamma * next_q_values
                else:
                    with torch.no_grad():
                        next_q_values = self.target_model(next_state_b).argmax(dim=1)
                        target_q_values = reward_b.flatten() + (1 - terminal_b.flatten()) * self.gamma * next_q_values

                loss = F.smooth_l1_loss(target_q_values, qsa_b)

                self.optimiser.zero_grad()     # Zero the gradients
                loss.backward()                # Backpropagation
                self.optimiser.step()          # Update the neural network parameters

            if step % self.update_frequency == 0:
                for target_network_param, q_network_param in zip(self.target_model.parameters(), self.model.parameters()):
                    target_network_param.data.copy_(
                        1.0 * q_network_param.data + (1.0 - 1.0) * target_network_param.data
                    )
                    

            if terminal:
                stats['Returns'].append(ep_reward)
                ep_reward = 0
                state, _ = env.reset()
                episode_count += 1

                # Epsilon Decay
                if self.epsilon > self.min_epsilon:
                    self.epsilon = self.epsilon - self.epsilon_decay

                
                # Save copies of training stages
                #if episode_count == 1:
                    self.model.save(filename=f"models/model_iter_{episode_count}.pt")

                if episode_count % 100 == 0:
                    print(f"Epsiode {episode_count} Average Returns {np.mean(stats['Returns'][-10:])} Epsilon {self.epsilon}")
                    self.model.save(filename=f"models/model_iter_{episode_count}.pt")

        return stats
    

    def test(self, env):
        for episode in range(5):
            total_reward = 0
            state, _ = env.reset()
            terminal = False
            for _ in range(self.max_step_ep):
                action = self.get_action(state)
                state, reward, terminal, _, _ = env.step(action)
                total_reward += reward
                if terminal:
                    break
            print(total_reward)

class random_Agent:
    def __init__(self, model, device='cpu', duelling = True, epsilon=1.0, min_epsilon=0.1, nb_actions=4, 
                 capacity=10000, batch_size = 32, learning_rate=0.00025, epsilon_decay = 0.05, 
                 gamma =0.99, max_step_ep = 2000, update_frequency = 1000, env_dims=None,
                 alpha = 0.2, beta = 0.5, beta_annealing = 0.000006):
        
        self.model = model
        self.nb_actions = nb_actions
        self.epsilon = epsilon
        
    def get_action(self, state):
        action = random.randint(0, self.nb_actions - 1)
        return action


    def train(self, env, total_steps):
        stats = {'Returns' : [], 'AvgReturns': [], 'EpsilonCheckPoint': []}
        ep_reward = 0
        state, _ = env.reset()
        episode_count = 0
        

        for step in range(1, total_steps + 1):
            
            action = self.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            terminal = terminated or truncated
            ep_reward += reward

            if terminal:
                stats['Returns'].append(ep_reward)
                ep_reward = 0
                state, _ = env.reset()
                episode_count += 1

                if episode_count % 100 == 0:
                    print(f"Epsiode {episode_count} Average Returns {np.mean(stats['Returns'][-10:])} Epsilon {self.epsilon}")       
                
        return stats