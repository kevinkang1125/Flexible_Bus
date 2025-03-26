import json
import numpy as np
import matplotlib.pyplot as plt


def load_json(file_path):
    """Load the JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def importance_sampling(trajectory, evaluation_policy,base_policy, up_b,low_b):
    """Calculate the importance weight and importance-weighted return for a trajectory."""
    action_list = trajectory["actions"]
    importance_weight = 1
    for j in range(len(action_list)):
        importance_weight *= (evaluation_policy[0][action_list[j][0]]*evaluation_policy[1][action_list[j][1]])/(base_policy[0][action_list[j][0]]*base_policy[1][action_list[j][1]])
    return_value  = trajectory["return"]
    norm_return = (return_value-low_b)/(up_b-low_b)
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
            (2 * np.log(2 / delta) / ((n - 1)*n*(len(data)-n))) *
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
        (np.log(2 / delta))*pairwise_sum / (n - 1)
    )/n
    lower_bound = term_1 - term_2 - term_3
    #lower_bound = term_3
    return best_c, lower_bound


return_list = np.loadtxt("return_list.txt")
file_path = "./trajectories.json"  # Replace with your file path
data = load_json(file_path)
evaluation_policy = [[0.9,0.1],[0.1,0.9]]
base_policy = [[0.5,0.5],[0.5,0.5]]
max_return = max(return_list)
min_return = min(return_list)
lower_bound_list = []
iters = []
for epochs in range(1000,30000,1000):
    random_sample = np.random.choice(data, size=epochs, replace=False)
    importance_sample = []
    print(epochs)
    for i in range(len(random_sample)):
        traj = random_sample[i]
        sample = importance_sampling(traj,evaluation_policy=evaluation_policy,base_policy = base_policy,up_b = max_return,low_b = min_return)
        importance_sample.append(sample)
    c_star, lower_bound = compute_lower_bound_bisection(importance_sample)
    lower_bound_list.append(lower_bound)
    iters.append(epochs)
print(c_star,lower_bound)
plt.plot(iters,lower_bound_list)
plt.title("90% Confidence Lower Bound for Time 1")
plt.xlabel("Num of Epochs")
plt.show()



