#!/usr/bin/env python3

from DAS_env import GazeboEnv
import numpy as np
from ddqn_PER import DDQNAgent
import matplotlib.pyplot as plt
import time 
import torch
import os 
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

file_name = "velodyne"
save_model = True
load_model = False
seed = 0 # Set random seed

# Create the network storage folders
'''if not os.path.exists("./DQN/results/ddqn"):
    os.makedirs("./DQN/results/ddqn")
if save_model and not os.path.exists("./DQN/pytorch_models/ddqn"):
    os.makedirs("./DQN/pytorch_models/ddqn")'''

writer = SummaryWriter("./DQN/runs/ddqn_PER/reward")

# Create the environment
#env = gym.make('CartPole-v1', render_mode='human')
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
n_states = environment_dim + robot_dim
n_actions = 5  # discret action (right/left/forward/f-l/f-r)

#Create DDQN-PER agent 
agent = DDQNAgent(n_states, n_actions, seed)

# Set the number of episodes and the maximum number of steps per episode
num_episodes = 1500
max_steps = 500

# Set the exploration rate
eps = eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

# Set the rewards and scores lists
rewards = []
scores = []

# Initialize success and collision tracking
success_count = 0
collision_count = 0

# Initialize variables for random obstacle strategy
random_near_obstacle = False  # Enable the strategy
count_rand_actions = 0  # Counter for random actions
random_action = None  # Placeholder for random discrete action


if load_model:
        try:
            agent.load(file_name, "./DQN/pytorch_models/ddqn")
            print("\033[32mloaded seccessflly\033[0m")
        except:
            print(
                "\033[31mCould not load the stored model parameters, initializing training with random parameters\033[0m"
            )

# Create the plot
plt.ion()  # Turn on interactive mode for real-time plotting
fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot([], [], label='DDQN_PER')
ax.set_xlim(0, num_episodes)
ax.set_ylim(-200, 200)  # Adjust based on your reward scale
ax.set_xlabel('Episode')
ax.set_ylabel('Score (Average Reward)')
ax.grid(True)
ax.legend()

# function for calculating exponential smoothing
def exponential_smoothing(data, alpha=0.95):
    smoothed_data = []
    last = data[0] 
    for point in data:
        smoothed_value = last * alpha + (1 - alpha) * point
        smoothed_data.append(smoothed_value)
        last = smoothed_value
    return smoothed_data

rewards_data = []

# Run the training loop
for i_episode in range(num_episodes):
#i_episode = 0
#while True: 
    print(f'Episode: {i_episode}')
    # Initialize the environment and the state
    state = env.reset()
    score = 0
    # eps = eps_end + (eps_start - eps_end) * np.exp(-i_episode / eps_decay)
    # Update the exploration rate
    eps = max(eps_end, eps_decay * eps)

    episode_success = False
    episode_collision = False
    
    # Run the episode
    for t in range(max_steps):

        # Check if the random_near_obstacle strategy should be applied
        '''if random_near_obstacle:
            if (
                np.random.uniform(0, 1) > 0.85
                and min(state[4:-8]) < 0.6  # Assuming state[4:-8] represents distance to obstacles
                and count_rand_actions < 1
            ):
                count_rand_actions = np.random.randint(8, 15)
                random_action = np.random.choice(np.arange(n_actions))  # Choose a random discrete action

            if count_rand_actions > 0:
                count_rand_actions -= 1
                action = random_action
            else:
                action = agent.act(state, eps)
        else:
            action = agent.act(state, eps)'''

        # Select an action and take a step in the environment
        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        # Store the experience in the replay buffer and learn from it
        agent.step(state, action, reward, next_state, done)
        # Update the state and the score
        state = next_state
        score += reward
         
        # Check for success or collision
        if done:
            if reward >= 130:  # Assuming +100 reward indicates success
                episode_success = True
            elif reward <= 0:  # Assuming -100 reward indicates collision
                episode_collision = True
            break
        
    if episode_success:
        success_count += 1
    if episode_collision:
        collision_count += 1
        
    print(f"\tScore: {score}, Epsilon: {eps}")

    writer.add_scalar('Cumulative Reward', score, i_episode)
    # Save the rewards and scores
    rewards.append(score)
    scores.append(np.mean(rewards[-100:]))
    rewards_data.append([i_episode, score])  # تخزين الحلقة والمكافأة في القائمة

    # احفظ البيانات في ملف Excel بعد كل 10 حلقات
    if i_episode % 10 == 0:
        df = pd.DataFrame(rewards_data, columns=["Episode", "Reward"])
        df.to_excel("./DQN/results/ddqn/reward_data.xlsx", index=False)  # حفظ الملف

     # smooth point using exponential smoothing
    smoothed_scores = exponential_smoothing(scores)

    # plot points in real-time 
    line.set_xdata(range(len(smoothed_scores)))
    line.set_ydata(smoothed_scores)
    ax.set_ylim(min(smoothed_scores)-10, max(smoothed_scores)+10)  # Dynamically adjust y-axis limits
    plt.draw()
    plt.pause(0.01)


      # Save the model after each episode
    if save_model and i_episode % 10 == 0:
        #agent.save2(i_episode, checkpoint_dir, eps)
        agent.save(file_name, directory="./DQN/pytorch_models/ddqn_PER")
    
    #i_episode += 1

# Calculate success and collision rates
success_rate = success_count / num_episodes
collision_rate = collision_count / num_episodes

# Print the results
print(f"\nSuccess Rate over {num_episodes} episodes: {success_rate * 100:.2f}%")
print(f"Collision Rate over {num_episodes} episodes: {collision_rate * 100:.2f}%")

plt.ioff()  # Turn off interactive mode
plt.show()  # Show final plot when the training is finished
