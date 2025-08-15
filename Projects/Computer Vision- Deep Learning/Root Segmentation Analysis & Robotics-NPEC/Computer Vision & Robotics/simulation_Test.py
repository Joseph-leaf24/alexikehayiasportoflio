from ot2_gym_wrapper import OT2Env
from stable_baselines3 import PPO
import wandb
import matplotlib.pyplot as plt
import time  # Import the time module
import numpy as np  # Import numpy for array manipulation

# Set a fixed goal position
fixed_goal = np.array([0.13105202, 0.08436922, 0.1691 ]) 

# Path to the locally stored model file
model_path = "C:/Users/User/Desktop/2024-25b-fai2-adsai-AlexiKehayias232230/datalab_tasks/task11/Local Runs/Iteration 4- Gamma-0.80/models/8xtnar2d/best_model.zip"

# Load the model artifact locally
run = wandb.init(mode="disabled")  # Disable wandb since we're loading locally
artifact = wandb.Artifact(name="local_model", type="model")
artifact.add_file(model_path)

# Extract and load the PPO model from the .wandb file
model = PPO.load(model_path)

# Create the environment
env = OT2Env(render=True, goals=fixed_goal)

# Reset the environment
obs, info = env.reset()

# Override the goal in the environment with the fixed goal
env.set_goal(fixed_goal)  # Assuming the environment has a `set_goal` method

# Initialize variables for reward, distance, and action tracking
reward_history = []  # Store total rewards for the current goal
distance_history = []  # Store distances per step
actions_history = []  # Store actions per step
episode_rewards = 0  # Cumulative reward for the current run
goal_steps = 0  # Count steps for this goal

# Run the model in the environment for n steps
n = 100000  # Replace with your desired number of steps
for i in range(n):
    # Predict the action using the model
    action, _states = model.predict(obs)
    
    # Take a step in the environment
    obs, r, term, trunc, info = env.step(action)
    
    # Accumulate rewards, distances, and actions
    episode_rewards += r
    distance = info.get('distance_to_goal', None)
    distance_history.append(distance)
    actions_history.append(action)
    
    # Log step information
    print(f"Step {i + 1}: Action = {action}, Reward = {r}, Distance = {distance}, Term = {term}, Trunc = {trunc}")
    
    # Add a delay to slow down the simulation
    time.sleep(0.1)
    
    # If term or trunc occurs, log the results and stop further processing
    if term or trunc:
        print(f"Terminating at step {i + 1}: Final Reward = {episode_rewards}, Final Distance = {distance}")
        reward_history.append(episode_rewards)
        break  # Stop processing when the goal is achieved or episode ends

# Plot the reward history
plt.figure(figsize=(10, 6))
plt.plot(reward_history, marker='o')
plt.xlabel('Goal (Single)')
plt.ylabel('Cumulative Reward')
plt.title('Model Performance for Single Goal')
plt.show()

# Plot the distance history
plt.figure(figsize=(10, 6))
plt.plot(distance_history)
plt.xlabel('Step')
plt.ylabel('Distance to Goal')
plt.title('Distance to Goal Over Steps')
plt.show()

# Output all actions taken
print("\nActions performed during the simulation:")
for step, action in enumerate(actions_history, 1):
    print(f"Step {step}: Action = {action}")
