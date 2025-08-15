import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000, stagnation_threshold=50, goals=None):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        self.stagnation_threshold = stagnation_threshold

        # Hardcoded working envelope bounds
        self.envelope = {
            "position_bounds": [[-0.1872, 0.2531], [-0.1711, 0.2201], [0.1691, 0.2896]]
        }

        # Predefined goals
        self.goals = goals or []  # List of specific goal positions
        self.goal_index = 0  # Current goal index

        # Create the simulation environment
        self.sim = Simulation(num_agents=1)

        # Define action space without velocity bounds (unrestricted actions)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Define observation space based on position bounds
        position_bounds = np.array(self.envelope["position_bounds"]).flatten()
        self.observation_space = spaces.Box(
            low=np.concatenate([position_bounds[::2], position_bounds[::2]]),
            high=np.concatenate([position_bounds[1::2], position_bounds[1::2]]),
            dtype=np.float32
        )

        self.steps = 0
        self.position_bounds = np.array(self.envelope["position_bounds"])

        # Track previous distance and position
        self.prev_distance_to_goal = None
        self.prev_pipette_position = None
        self.stagnation_steps = 0
        
    def get_plate_image(self):
        """
        Retrieves the current plate image path from the simulation.

        Returns:
            str: Path to the current plate image.
        """
        return self.sim.get_plate_image()
    
    def set_goal(self, goal):
        """
        Set a fixed goal for the environment.

        Args:
            goal (np.ndarray): The fixed goal position as a NumPy array.
        """
        if isinstance(goal, np.ndarray) and goal.shape == (3,):
            self.goal = goal
        else:
            raise ValueError("Goal must be a NumPy array with shape (3,).")

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        observation = self.sim.reset(num_agents=1)
        robot_key = next((key for key in observation if "pipette_position" in observation[key]), None)
        if robot_key is None:
            raise KeyError(f"No valid robot ID key found in observation. Got: {observation}")

        pipette_position = np.array(observation[robot_key]["pipette_position"])
        pipette_position = np.clip(pipette_position, self.position_bounds[:, 0], self.position_bounds[:, 1])

        # Use predefined goal
        if self.goals and self.goal_index < len(self.goals):
            self.goal_position = np.array(self.goals[self.goal_index]["tip"])  # Ensure goal_position is a numerical array
            self.goal_index = (self.goal_index + 1) % len(self.goals)  # Cycle through goals
        else:
            raise ValueError("No predefined goals available.")

        # Ensure the goal is far enough from the initial pipette position
        while np.linalg.norm(self.goal_position - pipette_position) < 0.01:
            raise ValueError("Goal position is too close to the initial pipette position.")

        obs = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)

        self.steps = 0
        self.prev_distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        self.prev_pipette_position = pipette_position
        self.stagnation_steps = 0

        # Validate observation space
        assert self.observation_space.contains(obs), "Initial observation out of bounds."

        return obs, {}

    def step(self, action):
        # Restrict the action values to the permitted boundaries
        action_with_drop = list(action) + [0]  # Append a placeholder for a hypothetical 'drop' action

        # Execute the simulation with the provided action and retrieve the output
        simulation_output = self.sim.run([action_with_drop])
        if not simulation_output:
            raise ValueError("Simulation execution did not produce any output.")
        if isinstance(simulation_output, list):
            observation = simulation_output[0]
        elif isinstance(simulation_output, dict):
            observation = simulation_output

        # Identify the key in the observation dictionary corresponding to the robot's state
        robot_key = next((key for key in observation if "pipette_position" in observation[key]), None)
        if robot_key is None:
            raise KeyError(f"No valid robot-related key found in the observation data. Received: {observation}")

        # Extract the pipette's position and ensure it stays within the specified bounds
        pipette_position = np.array(observation[robot_key]["pipette_position"])
        pipette_position = np.clip(pipette_position, self.position_bounds[:, 0], self.position_bounds[:, 1])

        # Combine the pipette's position and the goal position into a single observation array
        obs = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)

        # Reward: negative distance to the goal
        distance = np.linalg.norm(np.array(pipette_position) - np.array(self.goal_position))
        reward = -distance  # Base reward is negative of distance (encouraging goal achievement)

        # Update stagnation logic (avoid oscillations)
        if distance < self.prev_distance_to_goal:
            self.stagnation_steps = 0  # Reset stagnation if progress is made
        else:
            self.stagnation_steps += 1  # Increment stagnation steps if no progress

        # Penalize small oscillations or movements away from the goal
        if distance > self.prev_distance_to_goal:
            reward -= 0.1  # Small penalty if the robot moves away from the goal
        else:
            reward += 0.1  # Reward progress if the robot is getting closer

        # Large reward for staying within a very small threshold from the goal
        if distance < 0.001:  # If within 1mm of the goal, consider it close enough
            reward += 10  # Give a large reward for reaching the goal

        # Convergence: Check if the robot has stayed near the goal for several steps
        if distance < 0.001:  # Threshold for proximity to the goal
            self.goal_reached_counter += 1
        else:
            self.goal_reached_counter = 0  # Reset counter if not near goal

        # Provide a large reward if the robot has stayed close to the goal for multiple steps
        if self.goal_reached_counter > 10:  # Adjust this value as necessary (e.g., 10 steps near goal)
            reward += 50  # Large reward for staying near the goal for several steps
            print("Goal reached successfully!")

        # Update the environment's state variables to reflect the latest step
        self.steps += 1

        # Termination condition: end if maximum steps reached, goal reached, or stagnation detected
        terminated = (
            self.steps >= self.max_steps
            or distance < 0.001  # Goal proximity
            or self.stagnation_steps >= self.stagnation_threshold  # Stagnation threshold
        )
        truncated = False  # Modify as necessary for truncation-related logic

        # Additional info for tracking
        info = {"distance_to_goal": distance, "goal_reached_counter": self.goal_reached_counter}

        # Save current metrics for reference in the next step
        self.prev_distance_to_goal = distance
        self.prev_pipette_position = pipette_position

        # Return the updated observation, computed reward, and termination status
        return obs, reward, terminated, truncated, info




    def render(self, mode="human"):
        pass

    def close(self):
        self.sim.close()