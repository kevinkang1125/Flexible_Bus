import pandas as pd
import numpy as np
prob = [0.4,0.4]
deviate_1 = np.random.choice([0, 1], p=[1-prob[0], prob[0]])
deviate_2 = np.random.choice([0, 1], p=[1-prob[1], prob[1]])
action = [deviate_1,deviate_2]
print(action)
