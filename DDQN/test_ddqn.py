#!/usr/bin/env python3

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random 
import time
from DAS_env import GazeboEnv
import matplotlib.pyplot as plt

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

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.qnetwork_local = QNetwork(state_size, action_size)

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
        
    def load(self, filename, directory):
        self.qnetwork_local.load_state_dict(torch.load('%s/%s_ddqn_PER.pth' % (directory, filename)))

file_name = "velodyne"
seed = 0 # Set random seed

environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
n_states = environment_dim + robot_dim
n_actions = 5  # discret action (right/left/forward/f-l/f-r)

# Create the DDQN agent
agent = DDQNAgent(n_states, n_actions)

try:
    agent.load(file_name, "./pytorch_models/ddqn_PER")
    print("\033[32mloaded seccessflly\033[0m")
except:
    print("\033[31mCould not load the stored model parameters, initializing training with random parameters\033[0m") 

# Set the number of test episodes
num_test_episodes = 100
max_steps = 500

# Set exploration rate to 0 (no exploration during testing)
eps = 0.0

# Store test rewards and metrics
test_rewards = []
collisions = 0
successes = 0
episode_times = []


# Run the testing loop
for i_episode in range(num_test_episodes):
    print(f'Test Episode: {i_episode + 1}/{num_test_episodes}')
    
    # Initialize the environment and the state
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_collisions = 0

    # Start the timer
    start_time = time.time()

    
    # Run the episode
    for t in range(max_steps):

        episode_steps += 1

        # Select an action based on the policy (no exploration)
        action = agent.act(state, eps)
        
        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)
        
        # Update the state and accumulate the reward
        state = next_state
        episode_reward += reward
        
        if reward < 0:
            episode_collisions += 1

        # If the episode ends, break the loop
        if done:
            if reward > 130:  # Assume a large positive reward means reaching the goal
                successes += 1
            break
    
     # End the timer and calculate the time taken for this episode
    end_time = time.time()
    episode_time = end_time - start_time
    episode_times.append(episode_time)

    # Store the total reward for this episode
    test_rewards.append(episode_reward)
    print(f'Reward for episode {i_episode + 1}: {episode_reward}')
    print(f'Time taken for episode {i_episode + 1}: {episode_time:.2f} seconds')

    # Store the total reward and collisions for this episode
    test_rewards.append(episode_reward)
    collisions += episode_collisions
    print(f'Reward for episode {i_episode + 1}: {episode_reward}')

# Calculate success rate, collision rate, and average time to reach the goal
success_rate = successes / num_test_episodes
collision_rate = collisions / (num_test_episodes * max_steps)
average_time = np.mean(episode_times)

# Print the metrics
print(f'Success Rate: {success_rate * 100}%')
print(f'Collision Rate: {collision_rate * 100}%')
print(f'Average Time to Reach Goal: {average_time:.2f} seconds')

# Calculate and print the average reward across all test episodes
average_test_reward = np.mean(test_rewards)
print(f'Average Test Reward: {average_test_reward}')
