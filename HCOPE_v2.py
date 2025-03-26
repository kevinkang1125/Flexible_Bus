import json
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def importance_sampling(trajectory, evaluation_policy, base_policy, up_b, low_b):
    action_list = trajectory["actions"]
    importance_weight = np.prod([
        (evaluation_policy[0][a[0]] * evaluation_policy[1][a[1]]) / 
        (base_policy[0][a[0]] * base_policy[1][a[1]])
        for a in action_list
    ])
    return_value = trajectory["return"]
    norm_return = (return_value - low_b) / (up_b - low_b)
    return importance_weight * norm_return

def compute_lower_bound_bisection(data, confidence_level=0.9, tol=1e-3):
    """Optimize the threshold c using bisection search and compute the lower confidence bound."""
    delta = 1 - confidence_level

    # Compute importance-weighted returns
    sample_size = len(data) // 20
    random_sample = np.random.choice(data, size=sample_size, replace=False)
    iw_returns = random_sample
    n = len(random_sample)

    # Initialize search bounds for c
    c_min = 1
    c_max = 50
    best_c = c_min
    best_lower_bound = -float('inf')

    while (c_max - c_min) > tol:
        c_mid = (c_min + c_max) / 2

        # Truncate importance-weighted returns
        truncated_returns = np.minimum(iw_returns, c_mid)

        # Compute empirical mean and variance
        empirical_mean = np.mean(truncated_returns)
        empirical_variance = np.var(truncated_returns, ddof=1)

        # Apply Theorem 1
        term_1 = empirical_mean
        term_2 = 7 * c_mid * np.log(2 / delta) / (3 * (n - 1))
        term_3 = np.sqrt(
            (2 * np.log(2 / delta) / ((n - 1) * n * (len(data) - n))) *
            (n * np.sum((truncated_returns / c_mid) ** 2) - (np.sum(truncated_returns / c_mid)) ** 2)
        )
        lower_bound = term_1 - term_2 - term_3

        # Update the best threshold if the lower bound improves
        if lower_bound > best_lower_bound:
            best_lower_bound = lower_bound
            best_c = c_mid

        # Update search bounds
        if lower_bound > best_lower_bound:
            c_min = c_mid
        else:
            c_max = c_mid

    n = len(data)
    truncated_returns = np.minimum(data, best_c)

    # Compute empirical mean and variance
    empirical_mean = np.mean(truncated_returns)
    pairwise_sum = np.sum([(truncated_returns[i] - truncated_returns[j])**2 for i in range(n) for j in range(n)])

    # Apply Theorem 1
    term_1 = empirical_mean
    term_2 = 7 * best_c * np.log(2 / delta) / (3 * (n - 1))
    term_3 = np.sqrt(
        (np.log(2 / delta)) * pairwise_sum / (n - 1)
    ) / n
    lower_bound = term_1 - term_2 - term_3
    return best_c, lower_bound

return_list = np.loadtxt("return_list.txt", dtype=np.float32)
data = load_json("./trajectories.json")
evaluation_policy = [[0.9, 0.1], [0.1, 0.9]]
base_policy = [[0.5, 0.5], [0.5, 0.5]]
max_return, min_return = max(return_list), min(return_list)

lower_bound_list, iters = [], []
# Wrap the outer loop in tqdm to monitor progress
for epochs in tqdm(range(1000, 45000, 5000), desc="Processing Epochs"):
    # Progress bar for sampling
    random_sample = np.random.choice(data, size=epochs, replace=False)
    importance_sample = []

    # Add progress bar for multiprocessing pool
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
    lower_bound_list.append(lower_bound)
    iters.append(epochs)

print(c_star, lower_bound)
plt.plot(iters, lower_bound_list)
plt.title("90% Confidence Lower Bound")
plt.xlabel("Num of Epochs")
plt.show()


