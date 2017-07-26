import numpy as np
# グリッドサーチ、ランダムサーチ => ハイパーパラメーターを求める方法


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))
"""
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
"""


def cross_entropy_error(y, t):
    print("Cross Entropy: ")
    print("Y is ", y)
    print("T is ", t)
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
"""
def cross_entropy_error(y,t):
    if y.ndim == 1:
        t.reshape(1, t.size)
        y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

def cross_entropy_error(y,t):
    delta = 1e - 7
    return -np.sum(t * np.log(y + delta))
"""


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        print("#############################")
        print("Numerical Gradient function: ", "fxh1: ", fxh1, "fxh2: ", fxh2)
        print("(fxh1 - fxh2) / (2 * h) = ", (fxh1 - fxh2) / (2 * h))
        print("")

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad
"""
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    print("SIZE: ", x.size)
    print(x)

    for i in range(x.size):
        print(i)
        print(x[i])
        tmp = x[i]
        x[i] = float(tmp) + h
        fxh1 = f(x)

        x[i] = float(tmp) - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = tmp
    return grad
"""


class SimpleNet:
    count = 0
    def __init__(self):
        # self.W = np.random.randn(2,3)
        self.W = np.array([[0.47355232, 0.9977393, 0.84668094],[0.85557411, 0.03563661, 0.69422093]])

    def predict(self, x):
        print("Predict function: ")
        print("X is ", x)
        print("W is ")
        print(self.W)
        print("------------------------")
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        print("Loss function: ")
        print("Z: ", z, " ", "Y: ", y, " ", "LOSS: ", loss, " ")
        return loss


net = SimpleNet()   # net.w = np.array([[0.47355232, 0.9977393, 0.84668094],[0.85557411, 0.3563661, 0.69422093]])
x = np.array([0.6, 0.9])
p = net.predict(x)    # [ 1.05414809  0.63071653  1.1328074 ]
t = np.array([0,0,1])
# print('LOSS: ', net.loss(x, t))    # LOSS:  0.928068538748


def f(W):
    print("F function: ")
    print("W is ", W)
    print("------------------------")
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)