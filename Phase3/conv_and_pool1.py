import numpy as np
import tensorflow as tf

# 入力画像
image = [
    [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ]
]
x = np.array(image)

x_ = tf.placeholder(tf.float32, [None, 3, 3])
x_image = tf.reshape(x_, [1, 3, 3, 1])  # データ数, 画像の縦幅, 横幅, チャネル数

# weight
filter_image = [
    [0, 1],
    [2, 3]
]
f = np.array([filter_image])
f_ = tf.Variable(f, dtype=tf.float32)
f_reshape = tf.reshape(f_, [2, 2, 1, 1]) # フィルタの縦幅, 横幅, 入力チャネル, 出力チャネル

# convolution layer
# u = tf.nn.conv2d(x_image, f_reshape, strides=[1, 1, 1, 1], padding="VALID")
# u = tf.nn.conv2d(x_image, f_reshape, strides=[1, 1, 1, 1], padding="SAME")
u = tf.nn.conv2d(x_image, f_reshape, strides=[1, 2, 2, 1], padding="SAME")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

u_ = sess.run(u, feed_dict={x_: x})
print("u = ", u_)