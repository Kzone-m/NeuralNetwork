from layer import Relu, SoftmaxWithLoss, Affine
import numpy as np


"""データ1"""
x = np.array([[0.1, 0.8]])
w1 = np.array([[10, 7], [0.8, 6]])
b1 = np.array([[1, 1]])
w2 = np.array([[0.4, 30], [0.8, 0.2]])
b2 = np.array([[1, 1]])
t = np.array([[1, 0]])
learning_rate = 0.02


"""データ2"""
# npz_xd = np.load("data.npz")
# x = np.array([npz_xd["x"][0]])    # len(x) => 784
# t = np.array([[0,0,0,0,0,1,0,0,0,0]])    # len(t) => 10, t = npz_xd["d"][0]
# w1 = np.random.rand(784, 100)
# w2 = np.random.rand(100, 10)
# b1 = np.random.rand(1, 100)
# b2 = np.random.rand(1, 10)
# learning_rate = 0.01


"""レイヤー準備"""
affine1 = Affine(w1, b1)
relu = Relu()
affine2 = Affine(w2, b2)
error = SoftmaxWithLoss()


"""計算開始"""
for i in range(10):
    a1 = affine1.forward(x)
    z1 = relu.forward(a1)
    a2 = affine2.forward(z1)
    loss = error.forward(a2, t)
    dloss = error.backward()
    da2 = affine2.backward(dloss)
    dz1 = relu.backward(da2)
    da1 = affine1.backward(dz1)

    affine2.W = affine2.W - (affine2.dW * learning_rate)
    affine2.b = affine2.b - (affine2.db * learning_rate)

    affine1.W = affine1.W - (affine1.dW * learning_rate)
    affine1.b = affine1.b - (affine1.db * learning_rate)

    print("E: ", loss)


    print(affine2.dW)
    print(affine2.db)
    # print(affine1.dW)
    # print(affine1.db)