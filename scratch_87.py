import numpy as np

dataset = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])
X = dataset[:, 0:2]
y = dataset[:, 2]

print(X, y)