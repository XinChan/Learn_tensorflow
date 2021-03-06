import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = False)

# parameters
learning_rate = 0.01
training_epochs = 5
batch_size = 256
display_step = 1
examples_to_show = 10

n_input = 784

# tf graph input 
X = tf.placeholder("float", [None, n_input])

# hidden layer settings
n_hidden_1 = 256
n_hidden_2 = 128

weights = {
	'encoder_w1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
	'encoder_w2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
	'decoder_w1':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
	'decoder_w2':tf.Variable(tf.random_normal([n_hidden_1,n_input]))
}

biases = {
	'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),
	'decoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_b2':tf.Variable(tf.random_normal([n_input]))
}

# building the encoder
def encoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_w1']), biases['encoder_b1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_w2']), biases['encoder_b2']))
	return layer_2

# building the decoder
def decoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_w1']), biases['decoder_b1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_w2']), biases['decoder_b2']))
	return layer_2

# construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# prediction
y_prediction = decoder_op

# targets
y_true = X

# define loss 
cost = tf.reduce_mean(tf.pow(y_true - y_prediction, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# train
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	total_batch = int(mnist.train.num_examples/batch_size)
	for epoch in range(training_epochs):
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict={X:batch_xs})

		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch + 1), "cost = ", "{:.9f}".format(c))

	print("optimizer finished")

	encode_decode = sess.run(y_prediction, feed_dict ={X:mnist.test.images[:examples_to_show]})

	f,a = plt.subplots(2,10, figsize=(10,2))
	for i in range(examples_to_show):
		a[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
		a[1][i].imshow(np.reshape(encode_decode[i], (28,28)))
	plt.show()

















