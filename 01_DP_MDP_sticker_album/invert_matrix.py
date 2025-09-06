

import numpy as np

R = np.array([1, 1, 0])
P = np.array([
    [0.5, 0.5, 0],
    [0, 0.5, 0.5],
    [0, 0, 1],
])
gamma = 0.5
I = np.identity(3)

inv = np.linalg.inv(I - gamma * P)

solution = inv @ R
solution


import numpy as np

R = np.array([1, 1, 0])
P = np.array([
    [0.5, 0.5, 0],
    [0, 0.5, 0.5],
    [0, 0, 1],
])
gamma = 0.5
I = np.identity(3)

inv = np.linalg.inv(I - gamma * P)

solution = inv @ R
solution
