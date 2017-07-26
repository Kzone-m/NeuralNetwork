import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i]
        x[i] = float(tmp) + h
        fxh1 = f(x)

        x[i] = float(tmp) - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = tmp
    return grad


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def relu(a):
    return np.maximum(0, a)


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def sigmoid(a):
    expo = np.exp(-a)
    return 1 / (1 + expo)