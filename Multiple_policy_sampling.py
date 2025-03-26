import gym
import numpy as np
import flexible_bus
from policy import model_1, model_2  # Assuming these are CPU-based models
import json
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import tqdm  # For progress monitoring

gamma = 0.99

def collect_trajectory(seed):
    """Collect a single trajectory given a random seed."""
    np.random.seed(seed)
    trajectory = {
        "observations": [],
        "actions": [],
        "return": 0.0,
        "policy": None
    }

    env = gym.make('FlexibleBus-v0')
    obs = env.reset()
    r = 0
    done = False
    model = np.random.choice([0, 1])  # Randomly select model 0 or 1
    trajectory["policy"] = model

    if model == 0:
        while not done:
            trajectory["observations"].append(obs.tolist() if isinstance(obs, np.ndarray) else obs)
            action = model_1(obs)  # Assuming model_1 is a CPU-based function
            trajectory["actions"].append(action.tolist() if isinstance(action, np.ndarray) else action)
            obs, rewards, done, info = env.step(action)
            r = r * gamma + rewards
    else:
        while not done:
            trajectory["observations"].append(obs.tolist() if isinstance(obs, np.ndarray) else obs)
            action = model_2(obs)  # Assuming model_2 is a CPU-based function
            trajectory["actions"].append(action.tolist() if isinstance(action, np.ndarray) else action)
            obs, rewards, done, info = env.step(action)
            r = r * gamma + rewards

    trajectory["return"] = r
    return trajectory, r

# Parallelized trajectory collection
num_trajectories = 1000000
batch_size = 1000  # Adjust based on memory availability

all_traj = []
all_return = []

with ProcessPoolExecutor() as executor:
    for batch_start in tqdm.tqdm(range(0, num_trajectories, batch_size), desc="Processing batches"):
        seeds = range(batch_start, min(batch_start + batch_size, num_trajectories))
        results = executor.map(collect_trajectory, seeds)
        for traj, ret in results:
            all_traj.append(traj)
            all_return.append(ret)

# Save trajectories to JSON file
np.savetxt("return_list.txt", all_return)

def save_trajectories_to_json(trajectories, filepath):
    """Save trajectories to a JSON file."""
    def custom_serializer(obj):
        # Handle NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    with open(filepath, 'w') as f:
        json.dump(trajectories, f, indent=4, default=custom_serializer)

# Save collected data
save_trajectories_to_json(all_traj, "trajectories.json")

# Plot the histogram
plt.hist(all_return, bins=30, edgecolor='k', alpha=0.7)  # Adjust 'bins' as needed
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Raw Reward Data')
plt.show()