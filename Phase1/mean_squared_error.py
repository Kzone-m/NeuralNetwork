import numpy as np

z = np.array([2.9, 7.2, 1.6])
d = np.array([3.0, 8.0, 0.0])


def mean_squared_error(z, d):
    return np.sum((z-d)**2)

print(mean_squared_error(z,d))