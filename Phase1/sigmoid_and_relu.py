import numpy as np


def sigmoid(a):
    expo = np.exp(-a)
    return 1 / 1 + expo


def relu(a):
    return np.maximum(0, a)


def calc(ws, xs, bs, f):
    a = ws.dot(xs) + bs
    z = f(a)
    print(a, z)
    return a, z


"""sample1:"""
ws = np.array([[1,0,4.1], [2,3,-2]])
xs = np.array([2,5,4])
bs = np.array([1])
zs1 = calc(ws, xs, bs, sigmoid)
# [ 19.4  12. ] [ 1.          1.00000614]


"""sample2:"""
ws = np.array([[4,1,4], [2,3,2]])
xs = np.array([3,2,1])
bs = np.array([-5, -6])
zs2 = calc(ws, xs, bs, sigmoid)
# [13  8] [ 1.00000226  1.00033546]


"""sample3:"""
ws = np.array([[4,1,4], [2,3,2]])
xs = np.array([3,2,1])
bs = np.array([-2])    # np.array([-2, -2])
z3 = calc(ws, xs, bs, sigmoid)
# [16 12] [ 1.00000011  1.00000614]
z4 = calc(ws, xs, bs, relu)
# [16 12] [16 12]














