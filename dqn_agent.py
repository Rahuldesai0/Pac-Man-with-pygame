import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    def __init__(self, state_size=16, action_size=4):
        self.state_size = state_size
        self.action_size = action_size

        # Episode tracking (YOU WANTED THIS BACK)
        self.episode_rewards = []           # List of total reward per completed episode
        self.current_episode_reward = 0.0   # Running total for current episode

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.lr = 0.0005
        self.batch_size = 64
        self.target_update_freq = 100

        # Experience replay
        self.memory = deque(maxlen=100000)
        self.steps = 0

        # Device & models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        # Track episode reward
        self.current_episode_reward += reward
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0  # Reset for next episode

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).cpu().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Double DQN
        next_actions = self.model(next_states).argmax(1).unsqueeze(1)
        max_next_q = self.target_model(next_states).gather(1, next_actions).squeeze(1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Target network update
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target()