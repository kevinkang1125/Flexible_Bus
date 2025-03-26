import json
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

import gym
import numpy as np
import flexible_bus
from policy import model_1, model_2  
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

    while not done:
        trajectory["observations"].append(obs.tolist() if isinstance(obs, np.ndarray) else obs)
        action = model_1(obs)  # Assuming model_1 is a CPU-based function
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
    """Save trajectories to a JSON file with type preservation."""
    def custom_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, int):  # Ensure integers remain integers
            return obj
        return str(obj)
    
    with open(filepath, 'w') as f:
        json.dump(trajectories, f, indent=4, default=custom_serializer)

save_trajectories_to_json(all_traj, "trajectories.json")

# Plot the histogram
# plt.hist(all_return, bins=30, edgecolor='k', alpha=0.7)  # Adjust 'bins' as needed
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram of Raw Reward Data')
# plt.show()


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
def importance_sampling(trajectory, evaluation_policy, base_policy, up_b, low_b):
    actions = trajectory["actions"]  # Assume it's a list of lists
    importance_weights = [
        (evaluation_policy[0][a[0]] * evaluation_policy[1][a[1]]) /
        (base_policy[0][a[0]] * base_policy[1][a[1]])
        for a in actions
    ]
    importance_weight = np.prod(importance_weights)
    norm_return = (trajectory["return"] - low_b) / (up_b - low_b)
    return importance_weight * norm_return



def compute_lower_bound_bisection(data, confidence_level=0.9, tol=1e-3):
    delta = 1 - confidence_level
    sample_size = len(data) // 20
    random_sample = np.random.choice(data, size=sample_size, replace=False)
    iw_returns = random_sample
    n = len(random_sample)

    c_min, c_max = 1, 50
    best_c, best_lower_bound = c_min, -float('inf')

    while (c_max - c_min) > tol:
        c_mid = (c_min + c_max) / 2
        truncated_returns = np.minimum(iw_returns, c_mid)

        empirical_mean = np.mean(truncated_returns)
        empirical_variance = np.var(truncated_returns, ddof=1)

        term_1 = empirical_mean
        term_2 = 7 * c_mid * np.log(2 / delta) / (3 * (n - 1))
        term_3 = np.sqrt(
            (2 * np.log(2 / delta) / ((n - 1) * n * (len(data) - n))) *
            (n * np.sum((truncated_returns / c_mid) ** 2) - (np.sum(truncated_returns / c_mid)) ** 2)
        )
        lower_bound = term_1 - term_2 - term_3

        if lower_bound > best_lower_bound:
            best_lower_bound = lower_bound
            best_c = c_mid

        if lower_bound > best_lower_bound:
            c_min = c_mid
        else:
            c_max = c_mid
    n = len(data)
    truncated_returns = np.minimum(data, best_c)
    pairwise_sum = 2*(n**2)* np.var(truncated_returns, ddof=0)

    term_1 = np.mean(truncated_returns)
    term_2 = 7 * best_c * np.log(2 / delta) / (3 * (n - 1))
    term_3 = np.sqrt((np.log(2 / delta)) * pairwise_sum / (n - 1)) / n
    lower_bound = term_1 - term_2 - term_3
    return best_c, lower_bound

return_list = np.loadtxt("return_list.txt", dtype=np.float32)
data = load_json("./trajectories.json")
evaluation_policy = [[0.9, 0.1], [0.1, 0.9]]
base_policy = [[0.5, 0.5], [0.5, 0.5]]
max_return, min_return = max(return_list), min(return_list)

lower_bound_list, iters = [], []
tolerance_ratio = 0.0001

for epochs in tqdm(range(1000, 300000, 5000), desc="Processing Epochs"):
    indices = np.random.choice(len(data), size=epochs, replace=False)
    random_sample = [data[idx] for idx in indices]
    importance_sample = []

    with Pool() as pool:
        importance_sample = list(
            tqdm(
                pool.starmap(
                    importance_sampling,
                    [(traj, evaluation_policy, base_policy, max_return, min_return) for traj in random_sample]
                ),
                total=epochs,
                desc=f"Calculating Importance Sampling for {epochs} Epochs",
                leave=False
            )
        )

    c_star, lower_bound = compute_lower_bound_bisection(importance_sample)
    #lower_bound_list_ = lower_bound_list
    lower_bound_list.append(lower_bound)
    iters.append(epochs)
    if epochs >10000:
        # if np.abs(np.mean(lower_bound_list)-np.mean(lower_bound_list_)) < tolerance_ratio*np.mean(lower_bound_list_):
        #     break
        if np.var(lower_bound_list[-20:])<tolerance_ratio*np.mean(lower_bound_list[-20:]):
            break
        

print(c_star, lower_bound)
plt.plot(iters, lower_bound_list,linewidth = 10)
plt.title("90% Confidence Lower Bound", fontsize=60)
plt.xlabel("Num of Epochs", fontsize=60)
plt.ylabel("Lower Bound", fontsize=60)
plt.tick_params(axis='both', which='major', labelsize=60)
plt.show()