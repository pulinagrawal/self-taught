
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment Sparse Autoencoder
# ------------
# 
# Structured from `2_fullyconnected.ipynb`
# 
# The goal of this assignment is to train a sparse autoencoder network on MNIST Data and visulize its validation data reconstruction.

# In[1]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import re
import sys
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:

def save_model(model, filename, step):
    with open(filename+str(step)+'.pkl', 'wb') as mf:
        pickle.dump(model, mf)

def load_model(filename):
    with open(filename, 'rb') as mf:
        model = pickle.load(mf)
    return model


# In[4]:

feature_weights, feature_biases, _ = load_model("model414000.pkl")


# First we load the MNIST data

# In[5]:

data_set = input_data.read_data_sets('', False)
training_data = data_set.train
testing_data = data_set.test


# Checking  the data

# In[6]:

images_feed, labels_feed = training_data.next_batch(10000, False)
image_size = 28
num_labels = 10
np.min(images_feed)


# Do validation testing:
# - data as a flat matrix,
# 

# In[7]:

validation_data = data_set.validation
valid_batch, validation_labels = validation_data.next_batch(validation_data.num_examples)


# In[8]:

beta = 3
rho = .1
nHidden = 196
image_size = 28
batch_size = 128


# In[9]:

step = 0

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    batch_data, _ = training_data.next_batch(batch_size)
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    v_l = 20000
    while True:
        step += 1
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        # Generate a minibatch.
        batch_data, _ = training_data.next_batch(batch_size)
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data}
        _, l, feature_weights, feature_biases = session.run(
                                                  [optimizer, loss, weights_hidden1, biases_hidden1], 
                                                  feed_dict=feed_dict)
        if step%100000 == 0:
            fbf = open('feature_baises_' + str(step) + '.pkl', 'wb')
            pickle.dump(feature_biases, fbf)
            fbf.close()
            fwf = open('feature_weights_' + str(step) + '.pkl', 'wb')
            pickle.dump(feature_weights, fwf)
            fwf.close()
            
        if verify_validation[0]:
            _, l, v_l, valid_out_data = session.run(
                                            [optimizer, train_loss, valid_loss, valid_output_units],
                                            feed_dict=feed_dict)
            print("step", step, " \tTrain loss ", l, "\tValid loss", v_l)
            if v_l < verify_validation[0]:
                    save_model(model, 'model', step)
                    print("Found better validation. Looking for better...")
                    verify_validation = True, v_l, step
            if step > verify_validation[2]+100:
                break
                
        if step%500 == 0:
            prev_v_l = v_l
            _, l, v_l, valid_out_data = session.run(
                                            [optimizer, loss, valid_loss,valid_output_units],
                                            feed_dict=feed_dict)
            print("step", step, " \tTrain loss ", l, "\tValid loss", v_l)
            if prev_v_l < v_l:
                    save_model(model, 'model', step)
                    print("Looking for better validation now...")
                    verify_validation = True, v_l, step


# ## Training Softmax Classifier

# In[10]:

test_dataset, testing_labels = testing_data.next_batch(testing_data.num_examples)


# In[11]:

batch_size = 128
num_labels = 10
nHidden = 196
beta = 3
rho = .1
tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    weights_hidden1 = tf.constant(feature_weights)
    biases_hidden1 = tf.constant(feature_biases)
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_batch)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    logit_weights= tf.Variable(tf.truncated_normal([nHidden, num_labels]))
    logit_biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    hidden1 = tf.nn.sigmoid(tf.matmul(tf_train_dataset, weights_hidden1)  + biases_hidden1)
    logits = tf.matmul(hidden1, logit_weights) + logit_biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.05, global_step, 100000, 0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(
      tf.nn.sigmoid(tf.matmul(tf_valid_dataset, weights_hidden1) + biases_hidden1), logit_weights) + logit_biases)
    test_prediction = tf.nn.softmax(tf.matmul(
      tf.nn.sigmoid(tf.matmul(tf_test_dataset, weights_hidden1) + biases_hidden1), logit_weights) + logit_biases)


# In[12]:

def reformat(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return labels

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def save_data(weights, biases, weights_filename, biases_filename, step):
    fbf = open(biases_filename + str(step) + '.pkl', 'wb')
    pickle.dump(biases, fbf)
    fbf.close()
    fwf = open(weights_filename + str(step) + '.pkl', 'wb')
    pickle.dump(weights, fwf)
    fwf.close()


# In[14]:

def save_for_supervised():
    return save_data(weights, biases, 'out_weights_', 'out_biases_', step)

step = 0
valid_labels = reformat(validation_labels)
test_labels = reformat(testing_labels)
valid_acc = 0
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    while True:
        step += 1
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        batch_data, labels = training_data.next_batch(batch_size)
        batch_labels = reformat(labels)
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions, weights, biases = session.run([optimizer, loss, train_prediction,
                                                          logit_weights, logit_biases],
                                                         feed_dict=feed_dict)
        if step%500 == 0:
            prev_valid_acc = valid_acc
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, 
                                                          batch_labels))

            valid_acc = accuracy(valid_prediction.eval(), valid_labels)
            print("Validation accuracy: %.1f%%" % valid_acc)
            
            if prev_valid_acc > valid_acc:
                save_for_supervised()
                break
                
        if step%100000 == 0:
            save_for_supervised()
            
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(),
                                             test_labels))


# In[20]:

valid_acc


# In[27]:

prev_valid_acc


# In[30]:

save_data(weights, biases, 'out_weights', 'out_biases', step)


# In[ ]:



