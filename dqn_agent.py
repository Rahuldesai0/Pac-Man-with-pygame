import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import math

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
    def __init__(self, state_size=16, action_size=4, device=None,
                 action_repeat_prob=0.3):                 # <-- tuned
        self.state_size = state_size
        self.action_size = action_size

        # Episode tracking
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.episode_count = 0

        # Debug tracking
        self.total_training_steps = 0
        self.losses = deque(maxlen=10000)
        self.q_value_samples = deque(maxlen=10000)

        # Hyperparameters (changed for stability)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995                         # per episode decay

        # optimizer lr lowered for stability
        self.lr = 5e-5
        self.batch_size = 64

        # target network soft-update tau (smooth updates)
        self.tau = 0.005        # soft update coefficient
        self.target_update_freq = 1000  # hard-update fallback

        # Experience replay
        self.memory = deque(maxlen=100000)
        self.steps = 0

        # Minimal memory before starting training
        self.min_replay_size = 2000

        # Device & models
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target(hard=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-08)
        # Use Huber loss for stability
        self.loss_fn = nn.SmoothL1Loss()

        # action-repeat parameter (makes movement smooth but not too sticky)
        self.action_repeat_prob = action_repeat_prob

        print(f"🤖 Agent initialized | Device: {self.device} | LR: {self.lr} | batch: {self.batch_size}")

    def update_target(self, hard=False):
        if hard:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            # soft update: target = tau*local + (1-tau)*target
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        # store raw reward (environment-level). We'll clip during training.
        self.memory.append((state, action, reward, next_state, done))
        self.current_episode_reward += reward

        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)

            # Decay epsilon once per episode (faster decay than before)
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Logging summary
            avg10 = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else self.current_episode_reward
            avg_loss = np.mean(self.losses) if len(self.losses) else 0.0
            avg_q = np.mean(self.q_value_samples) if len(self.q_value_samples) else 0.0

            print(f"Ep {self.episode_count:4d} | R: {self.current_episode_reward:7.1f} | Avg10: {avg10:7.1f} | "
                  f"ε: {self.epsilon:.3f} | Loss: {avg_loss:.4f} | Q: {avg_q:6.3f} | Mem: {len(self.memory):5d}")

            if avg_loss > 50:
                print("   ⚠️  High loss detected (consider lowering lr or clipping rewards)")

            # reset per-episode accumulators
            self.current_episode_reward = 0.0

    def act(self, state):
        # Action repeat (makes movement stable) - tuned probability
        if hasattr(self, "last_action") and random.random() < self.action_repeat_prob:
            # still sample Q-values for diagnostics (without running policy update)
            # Append last predicted Q if available (best-effort)
            try:
                # compute Q for current state once for logging; do it sparingly
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.model(state_tensor)
                self.q_value_samples.append(float(q_values.max().cpu().numpy()))
            except Exception:
                pass
            return self.last_action

        if np.random.rand() < self.epsilon:
            action = random.randrange(self.action_size)
            # append an approximate Q sample to keep stats informative
            self.q_value_samples.append(0.0)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            qmax = float(q_values.max().cpu().numpy())
            self.q_value_samples.append(qmax)
            action = torch.argmax(q_values).cpu().item()

        self.last_action = action
        return action


    def replay(self):
        # wait until memory has enough samples
        if len(self.memory) < max(self.batch_size, self.min_replay_size):
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        # clip rewards to avoid huge target magnitudes (common stabilization trick)
        rewards = torch.FloatTensor(np.clip(np.array(rewards, dtype=np.float32), -20.0, 20.0)).to(self.device)
        dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).to(self.device)

        # Double DQN target calculation
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1).unsqueeze(1)          # actions by online network
            max_next_q = self.target_model(next_states).gather(1, next_actions).squeeze(1)  # values by target network
            targets = rewards + (1.0 - dones) * (self.gamma * max_next_q)

        # current Q-values for taken actions
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(q_values, targets)
        loss_value = float(loss.detach().cpu().item())
        self.losses.append(loss_value)
        self.total_training_steps += 1

        if math.isnan(loss_value):
            print(f"🚨 NaN loss at step {self.total_training_steps}, skipping update")
            return

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        # soft update target network each step
        self.update_target(hard=False)

        # book-keeping
        self.steps += 1
        # optional hard update fallback
        if self.steps % self.target_update_freq == 0:
            self.update_target(hard=True)

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
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.episode_count = checkpoint.get('episode_count', self.episode_count)
        self.episode_rewards = checkpoint.get('episode_rewards', self.episode_rewards)
        print(f"📂 Loaded from {filepath}")