import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load data from text file
import numpy as np
import matplotlib.pyplot as plt

# Load data from text file
data = np.loadtxt('importance_sample.txt', dtype=float)  # Replace 'data.txt' with your filename

# Plot the histogram
plt.hist(data, bins=30, edgecolor='k', alpha=0.7)  # Adjust 'bins' as needed

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Importance Sampled Data')

# Show the plot
plt.show()

