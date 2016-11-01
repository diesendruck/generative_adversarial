from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data
import tensorflow as tf
import pdb
from sbm_gan_utils import unique_arrays, next_n_from_gen
import math


def run_binary_classifier_nn(input_data):
    """
    Runs binary classifier neural network on data with labels.

    Args:
      input_data: A DataLabels named tuple with train and test data, each with
        corresponding labels.

    Returns:
      adjacency_matrix: A 2D array, representing the link network between the
        num_nodes nodes.
    """
    # Import data
    # mnist = mnist_input_data.read_data_sets('data', one_hot=True)
    NLABELS = len(unique_arrays(input_data.train.labels))
    INPUT_DIM = len(input_data.train.data[0])

    sess = tf.InteractiveSession()

    # Create the model
    graph_type = 'layers'
    if graph_type == 'simple':
        x = tf.placeholder(tf.float32, [None, INPUT_DIM], name='x-input')
        W = tf.Variable(tf.zeros([INPUT_DIM, NLABELS]), name='weights')
        b = tf.Variable(tf.zeros([NLABELS], name='bias'))
        y = tf.nn.softmax(tf.matmul(x, W) + b)
    elif graph_type == 'layers':
        x = tf.placeholder(tf.float32, [None, INPUT_DIM], name='x-input')
        hidden1_units = 128
        hidden2_units = 32
        with tf.name_scope('hidden1'):
            weights = tf.Variable(
                tf.truncated_normal([INPUT_DIM, hidden1_units],
                                    stddev=1.0 / math.sqrt(
                                        float(INPUT_DIM))),
                name='weights')
            biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)

        with tf.name_scope('hidden2'):
            weights = tf.Variable(
                tf.truncated_normal([hidden1_units, hidden2_units],
                                    stddev=1.0 / math.sqrt(
                                        float(hidden1_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(
                tf.truncated_normal([hidden2_units, NLABELS],
                                    stddev=1.0 / math.sqrt(
                                        float(hidden2_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([NLABELS]),
                                 name='biases')
            logits = tf.matmul(hidden2, weights) + biases

        y = tf.nn.softmax(logits)


    # Add summary ops to collect data
    # _ = tf.histogram_summary('weights', W)
    # _ = tf.histogram_summary('biases', b)
    # _ = tf.histogram_summary('y', y)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, NLABELS], name='y-input')

    # More name scopes will clean up the graph representation
    diff = tf.nn.softmax_cross_entropy_with_logits(y, y_)
    with tf.name_scope('cross_entropy'):
        # cross_entropy = -tf.reduce_mean(y_ * tf.log(y))
        cross_entropy = tf.reduce_mean(diff)
        _ = tf.scalar_summary('cross entropy', cross_entropy)
    with tf.name_scope('train'):
        option = 3
        if option == 1:
            train_step = tf.train.GradientDescentOptimizer(10.).minimize(
                cross_entropy)
        elif option == 2:
            train_step = tf.train.AdamOptimizer(0.1).minimize(
                cross_entropy)
        elif option == 3:
            train_step = tf.train.MomentumOptimizer(1., 0.9).minimize(
                cross_entropy)
    with tf.name_scope('test'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        _ = tf.scalar_summary('accuracy', accuracy)

    # Merge all the summaries and write them out to directory.
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('/tmp/sbm_classifier', sess.graph)
    tf.initialize_all_variables().run()

    # Train the model, and feed in test data and record summaries every 10 steps
    train_data_gen = (i for i in input_data.train.data)
    train_labels_gen = (i for i in input_data.train.labels)
    num_samples = len(input_data.train.data)
    batch_size = int(num_samples/1000)
    num_training_iterations = int(num_samples/batch_size)
    for i in range(num_training_iterations):
        if i % 10 == 0:  # Record summary data and the accuracy
            labels = input_data.test.labels
            feed = {x: input_data.test.data, y_: labels}

            result = sess.run([merged, accuracy, cross_entropy], feed_dict=feed)
            summary_str = result[0]
            acc = result[1]
            loss = result[2]
            writer.add_summary(summary_str, i)
            print('Accuracy at step %s: %s - loss: %f' % (i, acc, loss))
        else:
            # batch_xs, batch_ys = mnist.train.next_batch(100)
            # batch_ys = batch_ys[:, 0:NLABELS]
            batch_xs = next_n_from_gen(batch_size, train_data_gen)
            batch_ys = next_n_from_gen(batch_size, train_labels_gen)
            feed = {x: batch_xs, y_: batch_ys}
        sess.run(train_step, feed_dict=feed)