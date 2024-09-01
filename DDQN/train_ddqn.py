from DAS_env import GazeboEnv
import numpy as np
from ddqn import DDQNAgent
import matplotlib.pyplot as plt
import time 
import torch
import os 

file_name = "velodyne"
save_model = True
load_model = False
seed = 0 # Set random seed

# Create the network storage folders
if not os.path.exists("./DQN/results/ddqn"):
    os.makedirs("./DQN/results/ddqn")
if save_model and not os.path.exists("./DQN/pytorch_models/ddqn"):
    os.makedirs("./DQN/pytorch_models/ddqn")

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

# Create the DDQN agent
agent = DDQNAgent(n_states, n_actions, seed)

# Set the number of episodes and the maximum number of steps per episode
num_episodes = 1200
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
            agent.load(file_name, "/home/belabed/DRL-robot-navigation/DQN/pytorch_models/ddqn")
            print("\033[32mloaded seccessflly\033[0m")
        except:
            print(
                "\033[31mCould not load the stored model parameters, initializing training with random parameters\033[0m"
            ) 
# Run the training loop
for i_episode in range(num_episodes):
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
        if random_near_obstacle:
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
            action = agent.act(state, eps)

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
            if reward == 100:  # Assuming +100 reward indicates success
                episode_success = True
            elif reward == -100:  # Assuming -100 reward indicates collision
                episode_collision = True
            break
        
    if episode_success:
        success_count += 1
    if episode_collision:
        collision_count += 1
        
    print(f"\tScore: {score}, Epsilon: {eps}")
    # Save the rewards and scores
    rewards.append(score)
    scores.append(np.mean(rewards[-100:]))

      # Save the model after each episode
    if save_model:
        agent.save(file_name, directory="/home/belabed/DRL-robot-navigation/DQN/pytorch_models/ddqn")

# Calculate success and collision rates
success_rate = success_count / num_episodes
collision_rate = collision_count / num_episodes

# Print the results
print(f"\nSuccess Rate over {num_episodes} episodes: {success_rate * 100:.2f}%")
print(f"Collision Rate over {num_episodes} episodes: {collision_rate * 100:.2f}%")

# Plot the results
plt.figure(figsize=(10, 5))
plt.ylabel("Reward")
plt.xlabel("Episode")
plt.plot(range(len(rewards)), scores, label='Score')
plt.legend()
plt.show()