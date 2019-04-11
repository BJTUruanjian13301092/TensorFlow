import tensorflow as tf
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# read data from file
df = pd.read_csv("D:\\USC\\Working\\AI Research\\spambase.data", header=None)
spam_data = df.values

# separate feature and label
data_X = spam_data[:, :-1]
data_Y = spam_data[:, -1:]
feature_num = len(data_X[0])
label_num = len(data_Y[0])
sample_num = len(data_X)
print("Size of train_X: {}x{}".format(sample_num, feature_num))
print("Size of train_Y: {}x{}".format(sample_num, label_num))

# data set
train_X = tf.placeholder(tf.float32)
train_Y = tf.placeholder(tf.float32)

# training target
W = tf.Variable(tf.random_normal([feature_num, 1]))
b = tf.Variable(tf.random_normal([1, 1]))

z = tf.matmul(train_X, W) + b
h = tf.sigmoid(z)

# accuracy
correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(train_Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = - \
    tf.reduce_mean(train_Y * tf.log(tf.clip_by_value(h, 1e-10, 1.0)) +
                   (1-train_Y)*tf.log(tf.clip_by_value((1-h), 1e-10, 1.0)))

optimizer = tf.train.GradientDescentOptimizer(0.0001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

feed_dict = {train_X: data_X[:4000], train_Y: data_Y[:4000]}
feed_dict_test = {train_X: data_X[4000:], train_Y: data_Y[4000:]}
for step in range(100000):
    sess.run(train, feed_dict)
    if step % 1000 == 0:
        print("Loss is: {}".format(sess.run(loss, feed_dict)))
        print("Testing Accuracy:", sess.run(accuracy, feed_dict_test))

print("Training Finished.")