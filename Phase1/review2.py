import numpy as np
from review2_functions import relu, numerical_gradient, softmax, mean_squared_error, cross_entropy_error

# 1
zs_1 = np.array([[3], [5], [1]])

ws_2 = np.array([[1,1,4], [2,-1,0]])
bs_2 = np.array([[2], [-3]])
us_2 = ws_2.dot(zs_1) + bs_2
zs_2 = relu(us_2)

ws_3 = np.array([[6, -3], [0, 4], [2, -1]])
bs_3 = np.array([[0], [-1], [-4]])
us_3 = ws_3.dot(zs_2) + bs_3
zs_3 = relu(us_3)
"""
[[84]
 [ 0]
 [24]]
"""


# 2
z3_2 = np.array([[8.1],[0.4],[1.5]])
d_2 = np.array([[10.0],[0.0],[0.0]])
e_2 = mean_squared_error(z3_2, d_2)
"""
3.01
"""


# 3
u3_3 = np.array([[4.8], [-3.4], [8.2]])
z3_3 = softmax(u3_3)
"""
[[  3.22951782e-02]
 [  8.86998600e-06]
 [  9.67695952e-01]]
"""


# 4
z3_4 = np.array([[0.18],[0.79],[0.03]])
d_4 = np.array([[0],[1],[0]])
e_4 = cross_entropy_error(z3_4, d_4)
"""
0.235722206939
"""


# 5
import tensorflow as tf
x, d = np.array([[4,8,1]]), np.array([[2,5]])
x_, d_ = tf.placeholder(tf.float32, [1, 3]), tf.placeholder(tf.float32, [1, 2])
w, b = tf.Variable(tf.random_normal([3, 2], mean=0.0, stddev=0.05)), tf.Variable(tf.zeros([2]))

u = tf.matmul(x_, w) + b
loss = tf.reduce_mean(tf.square(u - d_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for step in range(200):
    print('step =', step, 'w = ', sess.run(w), 'b= ', sess.run(b))
    print('u', sess.run(u, feed_dict={x_: x, d_: d}), 'loss=', sess.run(loss, feed_dict={x_: x, d_: d}))
    sess.run(train_step, feed_dict={x_: x, d_: d})


# Extra
d = 10    # target
alpha = 0.01    # learning rate
ws = np.array([[0.1332, 0.7105]])    # initial weight
xs = np.array([[2], [3]])    # initial input
bs = np.array([[0]])    # initial bias

for i in range(200):
    y = ws.dot(xs) + bs    # output

    e = mean_squared_error(y, d)    # error

    tmp_ws = alpha * (y - d) * xs    # update value for weight
    tmp_bs = alpha * (y - d)     # update value for bias

    ws = ws - tmp_ws.T    # weight updated by subtracting update value for weight from previous weight
    bs = bs - tmp_bs    # bias updated by subtracting update value for bias from previous bias

    print("Step" + str(i+1) + ":")
    print("Output:", y)
    print("Error:", e)
    print("Weight: ", ws)
    print("Bias: ", bs)
    print()

"""
Step1:
Output: [[ 2.3979]]
Error: 28.895962205
Weight:  [[ 0.285242  0.938563]]
Bias:  [[ 0.076021]]

...

Step50:
Output: [[ 9.995308]]
Error: 1.10074160637e-05
Weight:  [[ 1.21863784  2.33865676]]
Bias:  [[ 0.54271892]]

...

Step100:
Output: [[ 9.99999751]]
Error: 3.10120522523e-12
Weight:  [[ 1.21921398  2.33952097]]
Bias:  [[ 0.54300699]]

...

Step200:
Output: [[10.]]
Error: 2.46164045474e-25
Weight:  [[ 1.21921429  2.33952143]]
Bias:  [[ 0.54300714]]
"""