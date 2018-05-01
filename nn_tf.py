import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import argparse
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# parameters
inputs, hiddens, outputs = 784, 200, 10
learning_rate = 0.01
epochs = 50
batch_size = 20

#loading the datasets
mnist = input_data.read_data_sets("./mnist/", one_hot=True)

# a random generator using uniform
def random(r, c, v):
    return tf.random_uniform([r,c], minval=-1/tf.sqrt(float(v)), maxval=1/tf.sqrt(float(v)))


# the neural network
def mlp(x, hidden_weights=None, output_weights=None):
    if hidden_weights == None:
        hidden_weights = tf.Variable(random(inputs, hiddens, inputs), name="hidden_weights")
    if output_weights == None:
        output_weights = tf.Variable(random(hiddens, outputs, hiddens), name="output_weights")
    hidden_outputs = tf.matmul(x, hidden_weights)
    hidden_outputs = tf.nn.sigmoid(hidden_outputs)  
    final_outputs = tf.matmul(hidden_outputs, output_weights)
    final_outputs = tf.nn.sigmoid(final_outputs)
    return final_outputs

# training with the train dataset
def train(x, y):
    final_outputs = mlp(x)
    errors = tf.reduce_mean(tf.squared_difference(final_outputs, y))
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(errors)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        total_batch = int(len(mnist.train.labels) / batch_size)
        for epoch in range(epochs):
            avg_error = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                _, c = sess.run([optimiser, errors], feed_dict={x: batch_x, y: batch_y})
                avg_error += c / total_batch
            print("Epoch [%d/%d], error: %.4f" %(epoch+1, epochs, avg_error))
        print("\nTraining complete!")
        saver.save(sess, "./model")
    
# predicting with the test dataset
def predict(x):    
    saver = tf.train.import_meta_graph("./model.meta")
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint("./"))
        graph = tf.get_default_graph()
        hidden_weights = graph.get_tensor_by_name("hidden_weights:0")
        output_weights = graph.get_tensor_by_name("output_weights:0")
        final_outputs = mlp(x, hidden_weights, output_weights)       
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(final_outputs, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)          
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [None, inputs])
    y = tf.placeholder(tf.float32, [None, outputs])       
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='predict' )
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.action == "predict":
        predict(x)
    if FLAGS.action == "train":
        train(x, y)