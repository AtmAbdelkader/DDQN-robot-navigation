"""
PRIORITIZED EXPIRIENCE REPLAY-BUFFER WITH DDQN
Author: Belabed Abdelkader
"""
#!/usr/bin/env python3 

import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Define the network architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 800)
        self.fc2 = nn.Linear(800, 600)
        self.fc3 = nn.Linear(600, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.max_priority = 1.0
        self.index = 0

    def push(self, state, action, reward, next_state, done):
        if state is None or next_state is None:
            return  

        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(float(self.max_priority))  # Ensure the priority is a float
        else:
            self.buffer[self.index] = experience
            self.priorities[self.index] = float(self.max_priority)  # Ensure the priority is a float

        self.index = (self.index + 1) % self.capacity



    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            print("Replay buffer is empty!")
            return None  

        # Convert priorities to a numpy array ensuring it contains only floats
        try:
            priorities = np.array([float(p) for p in self.priorities], dtype=np.float32)
        except ValueError as e:
            print("Error converting priorities to numpy array:", e)
            return None

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        if any(sample is None for sample in samples):
            print("Found None in samples!")
            return None  

        states, actions, rewards, next_states, dones = zip(*samples)

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights).float()

        return (
            torch.tensor(np.array(states)).float(),
            torch.tensor(np.array(actions)).long(),
            torch.tensor(np.array(rewards)).unsqueeze(1).float(),
            torch.tensor(np.array(next_states)).float(),
            torch.tensor(np.array(dones)).unsqueeze(1).int(),
            weights
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority)  # Ensure the priority is a float
            self.max_priority = max(self.max_priority, priority)


    def __len__(self):
        return len(self.buffer)

class DDQNAgent:
    def __init__(self, state_size, action_size, seed, learning_rate=0.001, capacity=1000000, 
                 discount_factor=0.99, tau=1e-3, update_every=4, batch_size=64, alpha=0.6, beta_start=0.4):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.steps = 0
        self.beta = beta_start
        self.beta_increment_per_sampling = 0.001

        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(capacity, alpha)
        #self.update_target_network()

        self.writer = SummaryWriter("<your path>/runs/ddqn_PER")

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)

        # Learn every update_every steps
        self.steps += 1
        if self.steps % self.update_every == 0 and len(self.replay_buffer) > self.batch_size:
            experiences = self.replay_buffer.sample(self.batch_size, beta=self.beta)
            self.learn(experiences)
            self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
    
    def act(self, state, eps=0.0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones, weights = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + self.discount_factor * (Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.view(-1, 1))

        # Compute TD error
        td_errors = torch.abs(Q_expected - Q_targets).detach().cpu().numpy()

        # Compute loss, apply importance-sampling weights
        loss = (F.mse_loss(Q_expected, Q_targets, reduction='none') * weights).mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in replay buffer
        batch_indices = np.arange(self.batch_size)
        self.replay_buffer.update_priorities(batch_indices, td_errors)

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        self.writer.add_scalar("Average Q-value", Q_expected.mean().item(), self.steps)
        self.writer.add_scalar("Max Q-value", Q_expected.max().item(), self.steps)
        self.writer.add_scalar("Loss", loss.item(), self.steps)

    def soft_update(self, local_model, target_model):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.qnetwork_local.state_dict(), '%s/%s_ddqn_PER.pth' % (directory, filename))

    def load(self, filename, directory):
        self.qnetwork_local.load_state_dict(torch.load('%s/%s_ddqn_PER.pth' % (directory, filename)))

