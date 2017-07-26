import csv
import tensorflow as tf

"""データの抽出"""
f = open('xs.csv', 'r')
reader = csv.reader(f)
xs = []  # 特徴量 14個
for row in reader:
    xs.append(list(map(float, row)))

f = open('ys.csv', 'r')
reader = csv.reader(f)
ys = []  # 各選手の打率
for row in reader:
    ys.append(list(map(float, row)))

"""特徴量(打席数,安打数,etc...)を入れる箱を準備"""
x = tf.placeholder(tf.float32, shape=[None, 14])  # (1, 14)

"""正解の値(打率)を入れる箱を準備"""
y_ = tf.placeholder(tf.float32, shape=[None, 1])  # (1, 1)

"""1層目のパラメーターの準備"""
w1 = tf.Variable(tf.truncated_normal([14, 10], stddev=0.1))  # (14, 10)
b1 = tf.Variable(tf.constant(1.0, shape=[10]))  # (1, 10)

"""2層目のパラメーターの準備"""
w2 = tf.Variable(tf.truncated_normal([10, 1], stddev=0.1))  # (10, 1)
b2 = tf.Variable(tf.constant(1.0, shape=[1]))  # (1, 1)

"""1層目の計算"""
a1 = tf.matmul(x, w1) + b1  # (1, 14)・(14, 10) + (1, 10)
z1 = tf.sigmoid(a1)  # => (1, 10)
# z1 = tf.nn.relu(a1)

"""2層目の計算"""
a2 = tf.matmul(z1, w2) + b2  # (1, 10)・(10, 1) + (1, 1)
y = tf.nn.relu(tf.matmul(z1, w2) + b2)  # (1, 1)

"""誤差の算出と逆伝搬"""
loss = tf.reduce_sum(tf.square(y - y_))  # 二乗誤差で実際の値との差を算出
train_step = tf.train.AdagradOptimizer(0.01).minimize(loss)  # アダムをしようして誤差を最小化するように学習

"""学習開始"""
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for s in range(500):
    for i in range(30):
        print(sess.run(loss, feed_dict={x: xs[i:i+1], y_: ys[i:i+1]}))
        print(sess.run(y, feed_dict={x: xs[i:i+1], y_: ys[i:i+1]}))
        sess.run(train_step, feed_dict={x: xs[i:i+1], y_: ys[i:i+1]})

print("実際の打率: ", ys[29:30][0][0])
print("予想の打率: ", sess.run(y[0][0], feed_dict={x: xs[29:30], y_: ys[29:30]}))
print("今回の誤差: ", sess.run(loss, feed_dict={x: xs[29:30], y_: ys[29:30]}))
"""
学習回数: 15000
実際の打率:  0.229
予想の打率:  0.239989
今回の誤差:  0.000120755
"""