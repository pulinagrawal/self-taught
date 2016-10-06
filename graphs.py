import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.examples.tutorials.mnist import input_data

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_valid_dataset = tf.constant(valid_batch)
    # tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights_hidden1 = tf.Variable(tf.truncated_normal([image_size * image_size, nHidden], stddev=0.01))
    weights = tf.Variable(tf.truncated_normal([nHidden, image_size * image_size], stddev=0.01))
    biases_hidden1 = tf.Variable(tf.zeros([nHidden]))
    biases = tf.Variable(tf.zeros([image_size * image_size]))

    # Training computation.
    hidden_comp = tf.matmul(tf_train_dataset, weights_hidden1)
    hidden1 = tf.nn.sigmoid(tf.mul(hidden_comp + biases_hidden1, 8))
    output_units = tf.nn.sigmoid(tf.matmul(hidden1, weights) + biases)

    # Sparsity computation
    int_rho = tf.reduce_sum(hidden1, 0)
    rho_hat = tf.div(int_rho, batch_size)
    rho_hat_mean = tf.reduce_mean(rho_hat)
    rho_in = tf.sub(tf.constant(1.), rho)
    rho_hat_in = tf.sub(tf.constant(1.), rho_hat)
    klterm = tf.add(tf.mul(rho, tf.log(tf.div(rho, rho_hat))),
                    tf.mul(rho_in, tf.log(tf.div(rho_in, rho_hat_in))))
    kl_div = tf.reduce_sum(klterm)

    loss = tf.div(tf.nn.l2_loss(tf.sub(output_units, tf_train_dataset)),
                  tf.constant(float(batch_size))) + beta * kl_div

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # Predictions for the training, validation, and test data.
    valid_output_units = tf.nn.sigmoid(tf.matmul(
        tf.nn.sigmoid(tf.mul(tf.matmul(tf_valid_dataset,
                                       weights_hidden1)
                             + biases_hidden1,
                             8)),
        weights) + biases)
    valid_loss = tf.div(tf.nn.l2_loss(
        tf.sub(valid_output_units, tf_valid_dataset)),
        tf.constant(float(batch_size)))
