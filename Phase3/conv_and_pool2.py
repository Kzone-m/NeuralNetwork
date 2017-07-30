import numpy
import tensorflow as tf

# load data from npz file
npz_xd = numpy.load("data.npz")
print('type(npz_xd)' + str(type(npz_xd)))
print('type(npz_xd["x"])=' + str(type(npz_xd["x"])))
print('type(npz_xd["d"])=' + str(type(npz_xd["d"])))
print('npz_xd["x"].shape=' + str(npz_xd["x"].shape))
print('npz_xd["d"].shape=' + str(npz_xd["d"].shape))
print('npz_xd["x"][0].shape=' + str(npz_xd["x"][0].shape))
print('npz_xd["x"][0][0]=' + str(npz_xd["x"][0][0]))
print('npz_xd["d"][0]=' + str(npz_xd["d"][0]))

# make input data, target data
x = npz_xd["x"]
d = npz_xd["d"]

# change target data from [N, 1] to [N, 10]
list_d = []
for data in d:
    if data == 0:
        list_d.append([1,0,0,0,0,0,0,0,0,0])
    elif data == 1:
        list_d.append([0,1,0,0,0,0,0,0,0,0])
    elif data == 2:
        list_d.append([0,0,1,0,0,0,0,0,0,0])
    elif data == 3:
        list_d.append([0,0,0,1,0,0,0,0,0,0])
    elif data == 4:
        list_d.append([0,0,0,0,1,0,0,0,0,0])
    elif data == 5:
        list_d.append([0,0,0,0,0,1,0,0,0,0])
    elif data == 6:
        list_d.append([0,0,0,0,0,0,1,0,0,0])
    elif data == 7:
        list_d.append([0,0,0,0,0,0,0,1,0,0])
    elif data == 8:
        list_d.append([0,0,0,0,0,0,0,0,1,0])
    elif data == 9:
        list_d.append([0,0,0,0,0,0,0,0,0,1])
d_mod = numpy.array(list_d)
print("d_mod.shape = " + str(d_mod.shape))
print("d_mod = " + str(d_mod))

x_train = x[0:5000]
x_test = x[5000:6000]
d_train = d_mod[0:5000]
d_test = d_mod[5000:6000]

# computation graph
x_ = tf.placeholder(tf.float32, [None, 784])
x_reshape = tf.reshape(x_, [-1, 28, 28, 1])
d_ = tf.placeholder(tf.float32, [None, 10])

w2 = tf.Variable(tf.random_normal([4, 4, 1, 15], mean=0.0, stddev=0.05), dtype=tf.float32)
b2 = tf.Variable(tf.zeros([15]), dtype=tf.float32)

z1 = x_reshape
# u2 = tf.matmul(z1, w2) + b2
u2 = tf.nn.conv2d(z1, w2, strides=[1, 3, 3, 1], padding="VALID") + b2
z2 = tf.sigmoid(u2)
z2_pool = tf.nn.max_pool(z2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

w3 = tf.Variable(tf.random_normal([375,10], mean=0.0, stddev=0.05), dtype=tf.float32)
b3 = tf.Variable(tf.zeros([10]), dtype=tf.float32)

z2_reshape = tf.reshape(z2_pool, [-1, 375])
u3 = tf.matmul(z2_reshape, w3) + b3
z3 = tf.nn.softmax(u3)

loss = -tf.reduce_mean(d_ * tf.log(z3))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

correct_prediction = tf.equal(tf.argmax(z3, 1), tf.argmax(d_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# make minibatch
def make_minibatch(per_list, x_data, d_data):
    x_data_mini = x_data[per_list]
    d_data_mini = d_data[per_list]
    return x_data_mini, d_data_mini

batch_size = 100

for epoch in range(10):
    
    x_random_list = numpy.random.permutation(len(x_train))
    for batch_count in range(len(x_train) // batch_size):
        
        tmp_list = x_random_list[batch_count * batch_size : (batch_count + 1) * batch_size]
        x_minibatch, d_minibatch = make_minibatch(tmp_list, x_train, d_train)
        
        sess.run(train_step, feed_dict={x_: x_minibatch, d_: d_minibatch})
        loss_ = sess.run(loss, feed_dict={x_: x_minibatch, d_: d_minibatch})
        print('epoch = ' + str(epoch) + ',training loss = ' + str(loss_))

    #test phase
    if epoch % 100 == 0:
        
        loss_test = sess.run(loss, feed_dict={x_: x_test, d_: d_test})
        print('accuracy', sess.run(accuracy, feed_dict={x_: x_test, d_: d_test}))
        print('test loss = ' + str(loss_test))


