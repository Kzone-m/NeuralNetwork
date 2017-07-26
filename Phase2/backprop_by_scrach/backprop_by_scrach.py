from functions import relu, softmax, cross_entropy_error
import numpy as np

"""データ"""
x = np.array([[0.1, 0.8]])
w1 = np.array([[10, 7], [0.8, 6]])
b1 = np.array([[1, 1]])
w2 = np.array([[0.4, 30], [0.8, 0.2]])
b2 = np.array([[1, 1]])
t = np.array([[1, 0]])
learning_rate = 0.02

for i in range(1):
    a1 = np.dot(x, w1) + b1
    z1 = relu(a1)
    a2 = np.dot(z1, w2) + b2
    y = softmax(a2)
    loss = cross_entropy_error(y, t)

    """共通の偏微分"""
    dEdY = - (t / y)  # [dEdy1, dEdy2]
    S = np.sum(np.exp(a2))
    dYdS = np.exp(a2) / np.square(S)    # [dY1dS, dY2dS]

    print("dw2_11: ", (-(t[0][0] / y[0][0]) * (np.exp(a2[0][1]) / np.square(S))) * np.exp(a2[0][0]) * z1[0][0])
    print("dw2_12: ", (-(t[0][0] / y[0][0]) * (np.exp(a2[0][1]) / np.square(S))) * np.exp(a2[0][1]) * z1[0][0])
    print("dw2_21: ", (-(t[0][0] / y[0][0]) * (np.exp(a2[0][1]) / np.square(S))) * np.exp(a2[0][0]) * z1[0][1])
    print("dw2_22: ", (-(t[0][0] / y[0][0]) * (np.exp(a2[0][1]) / np.square(S))) * np.exp(a2[0][1]) * z1[0][1])
    print("db2_1: ", (-(t[0][0] / y[0][0]) * (np.exp(a2[0][1]) / np.square(S))) * np.exp(a2[0][0]))
    print("db2_2: ",(-(t[0][0] / y[0][0]) * (np.exp(a2[0][1]) / np.square(S))) * np.exp(a2[0][1]))

    """ 0が掛けられている式を省略しないで書いた形
    print("dw2_11: ", (((t[0][0] / y[0][0]) * (np.exp(a2[0][1]) / np.square(S))) + (-(t[0][1] / y[0][1]) * (np.exp(a2[0][0]) / np.square(S)))) * np.exp(a2[0][0]) * z1[0][0])
    print("dw2_12: ", (((t[0][0] / y[0][0]) * (np.exp(a2[0][1]) / np.square(S))) + (-(t[0][1] / y[0][1]) * (np.exp(a2[0][0]) / np.square(S)))) * np.exp(a2[0][1]) * z1[0][0])
    print("dw2_21: ", (((t[0][0] / y[0][0]) * (np.exp(a2[0][1]) / np.square(S))) + (-(t[0][1] / y[0][1]) * (np.exp(a2[0][0]) / np.square(S)))) * np.exp(a2[0][0]) * z1[0][1])
    print("dw2_22: ", (((t[0][0] / y[0][0]) * (np.exp(a2[0][1]) / np.square(S))) + (-(t[0][1] / y[0][1]) * (np.exp(a2[0][0]) / np.square(S)))) * np.exp(a2[0][1]) * z1[0][1])
    """

"""  
dW2の結果:
    dw2_11: -2.64,   dw2_12: -4.62768128995e+32
    dw2_21: -6.5,    dw2_22: -1.13939122669e+33

dW2の結果(正解):
    dw2_11: -2.64,   dw2_12: 2.64
    dw2_21: -6.5,    dw2_22: 6.5


dB2の結果:
    db2_1:  -1.0,  db2_2:  -1.75290957953e+32
    
dB2の正解:
    db2_1:  -1,  db2_2: 1
"""