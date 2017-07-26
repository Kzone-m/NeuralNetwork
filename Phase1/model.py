import numpy as np
from review2_functions import relu, numerical_gradient, softmax, mean_squared_error, cross_entropy_error, sigmoid


"""初期設定"""
d = 1    # target
alpha = 0.05    # learning rate
n = 10000  # iteration number

"""パラメーター"""
ws = np.array([[0.1332, 0.7105]])    # initial weight
bs = np.array([[0]])    # initial bias

"""入力"""
xs = np.array([[2], [3]])    # initial input


for i in range(n):
    y = sigmoid(ws.dot(xs) + bs)   # output
    e = mean_squared_error(y, d)    # error

    dEdy = y - d
    dydw = (1 - y) * y * xs
    dydb = (1 - y) * y

    # dydw = ( > 0).astype(np.float32)
    # dydb = (b2 > 0).astype(np.float32)

    tmp_ws = alpha * dEdy * dydw    # update value for weight
    tmp_bs = alpha * dEdy * dydb    # update value for bias

    ws = ws - tmp_ws.T    # weight updated by subtracting update value for weight from previous weight
    bs = bs - tmp_bs    # bias updated by subtracting update value for bias from previous bias

    print("Step" + str(i+1) + ":")
    print("Output:", y)
    print("Error:", e)
    print("dEdy", dEdy)
    print("dydw", dydw)
    print("Weight: ", ws)
    print("Bias: ", bs)
    print()
