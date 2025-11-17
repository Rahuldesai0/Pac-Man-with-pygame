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

        # Episode tracking
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.episode_count = 0
        
        # Debug tracking
        self.total_training_steps = 0
        self.losses = []
        self.q_value_samples = []

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
        
        print(f"🤖 Agent initialized | Device: {self.device}")

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.current_episode_reward += reward
        
        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            
            # Decay epsilon ONCE per episode (not per step!)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Compact episode summary
            avg_10 = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else self.current_episode_reward
            avg_loss = np.mean(self.losses[-100:]) if len(self.losses) >= 100 else (np.mean(self.losses) if self.losses else 0)
            avg_q = np.mean(self.q_value_samples[-100:]) if len(self.q_value_samples) >= 100 else (np.mean(self.q_value_samples) if self.q_value_samples else 0)
            
            print(f"Ep {self.episode_count:4d} | R: {self.current_episode_reward:7.1f} | "
                  f"Avg10: {avg_10:7.1f} | ε: {self.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f} | Q: {avg_q:6.1f} | Mem: {len(self.memory):5d}")
            
            # Warnings only when needed
            if avg_loss > 10:
                print(f"   ⚠️  High loss detected")
            if len(self.memory) < 1000 and self.episode_count > 10:
                print(f"   ⚠️  Low memory size - may not be learning")
            
            self.current_episode_reward = 0.0

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        self.q_value_samples.append(q_values.max().item())
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
        
        self.losses.append(loss.item())
        self.total_training_steps += 1
        
        # Critical error check
        if torch.isnan(loss):
            print(f"🚨 NaN loss at step {self.total_training_steps}!")
            return
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Epsilon decay removed from here - now happens once per episode in remember()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target()
    
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'episode_rewards': self.episode_rewards,
        }, filepath)
        print(f"💾 Saved to {filepath}")
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.episode_rewards = checkpoint['episode_rewards']
        print(f"📂 Loaded from {filepath}")