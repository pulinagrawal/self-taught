
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

# In[ ]:

from utils import vae


# In[1]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import re
import sys
import pickle

import numpy as np
import tensorflow as tf
#from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
#from tensorflow.examples.tutorials.mnist import input_data
from mnist import read_data_sets


# In[2]:


class NoValue:
    def __init__(self):
        pass
    def __getitem__(self):
        pass
    
_no_value = NoValue()



# In[3]:

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
    
def save_model(model, filename, step):
    with open(filename+str(step)+'.pkl', 'wb') as mf:
        pickle.dump(model, mf)

def load_model(filename):
    with open(filename, 'rb') as mf:
        model = pickle.load(mf)
    return model


# First we load the MNIST data

# ##### Self data extracted but never used.

# In[4]:

data_set, self_data_set = read_data_sets('', False)
self_data = self_data_set.train
training_data = data_set.train
validation_data = data_set.validation
testing_data = data_set.test


# Do validation testing:
# - data as a flat matrix,
# 

# In[5]:

print(self_data._num_examples)
print(training_data.num_examples)
print(validation_data.num_examples)
print(testing_data.num_examples)


# In[17]:

validation_data = data_set.validation
valid_batch, validation_labels = validation_data.next_batch(validation_data.num_examples)


# In[ ]:

network_architecture =     dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=20)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=75)


# In[18]:

beta = 3
rho = .1
nHidden = 196
image_size = 28
batch_size = 128

start_model = _no_value
sparse = False
batch_size, input_size, nHidden, valid_batch = (batch_size, image_size*image_size, 
                                                                         nHidden, 
                                                                         valid_batch
                                                                    )
vae = VariationalAutoencoder(network_architecture, 
                             learning_rate=learning_rate, 
                             batch_size=batch_size)


# In[ ]:

def stopping_criterion(curr_valid, best_valid):
    if curr_valid[0] < best_valid[0]:
        False
    else:
        if curr_valid[1] > best_valid[1]+100:
            True
        else:
            False


# In[26]:


with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    step = 0
    verify_validation = False, 1, 1
    v_l = 20000
    best_validation = 10000, 0, start_model
    strict_checking = False
    
    print("Initialized")
    batch_data, _ = self_data.next_batch(batch_size)
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    while True:
        step += 1
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        # Generate a minibatch.
        batch_data, _ = self_data.next_batch(batch_size)
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data}
        out = session.run(
                                                  [optimizer, loss, learning_rate, weights_hidden1, biases_hidden1, biases], 
                                                  feed_dict=feed_dict)
        _, l, learn_rate, model = out[0], out[1], out[2], out[3:]
        if step%100000 == 0:
            save_model(model, 'model-self', step)

            
        if step%500 == 0:
            prev_v_l = v_l
            _, l, v_l, valid_out_data = session.run(
                                    [optimizer, loss, valid_loss, valid_output_units],
                                            feed_dict=feed_dict)
            print("step", step, " \tTrain loss ", l, "\tValid loss", v_l, "\tLearning Rate", learn_rate)
            if v_l > prev_v_l:
                strict_checking = True
           
        if strict_checking:
            _, l, v_l, valid_out_data = session.run(
                                    [optimizer, loss, valid_loss, valid_output_units],
                                            feed_dict=feed_dict)
        
            if stopping_criterion((v_l, step, model), best_validation):
                save_model(best_validation[2], 'model-self', best_validation[1])
                break
            else:
                if v_l < best_validation[0]:
                    best_validation = (v_l, step, model)
    
    print("Tuning")
    step = 0
    verify_validation = False, 1, 1
    v_l = 20000
    best_validation = 10000, 0, start_model
    strict_checking = False
    
    batch_data, _ = training_data.next_batch(batch_size)
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
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
        out = session.run(
                                                  [optimizer, loss, learning_rate, weights_hidden1, biases_hidden1, biases], 
                                                  feed_dict=feed_dict)
        _, l, learn_rate, model = out[0], out[1], out[2], out[3:]
        if step%100000 == 0:
            save_model(model, 'model-tuning', step)

            
        if step%500 == 0:
            prev_v_l = v_l
            _, l, v_l, valid_out_data = session.run(
                                    [optimizer, loss, valid_loss, valid_output_units],
                                            feed_dict=feed_dict)
            print("step", step, " \tTrain loss ", l, "\tValid loss", v_l, "\tLearning Rate", learn_rate)
            if v_l > prev_v_l:
                strict_checking = True
           
        if strict_checking:
            _, l, v_l, valid_out_data = session.run(
                                    [optimizer, loss, valid_loss, valid_output_units],
                                            feed_dict=feed_dict)
        
            if stopping_criterion((v_l, step, model), best_validation):
                save_model(best_validation[2], 'model-tuning', best_validation[1])
                break
            else:
                if v_l < best_validation[0]:
                    best_validation = (v_l, step, model)


# In[4]:

feature_weights, feature_biases, biases = load_model('model414000.pkl')


# ### Output from the Run on cluster

# .
# .
# .
# .
# 
# step 9750000    Train loss  2.56298     Valid loss 113.472
# 
# step 9750500    Train loss  2.80224     Valid loss 113.468
# 
# step 9751000    Train loss  2.59853     Valid loss 113.471
# step 9751500    Train loss  2.65598     Valid loss 113.47
# step 9752000    Train loss  2.50461     Valid loss 113.47
# step 9752500    Train loss  2.63187     Valid loss 113.468
# step 9753000    Train loss  2.84001     Valid loss 113.472
# step 9753500    Train loss  2.6863      Valid loss 113.471
# step 9754000    Train loss  2.73344     Valid loss 113.47
# step 9754500    Train loss  2.63237     Valid loss 113.47
# step 9755000    Train loss  2.52693     Valid loss 113.47
# step 9755500    Train loss  2.56478     Valid loss 113.469
# step 9756000    Train loss  2.50033     Valid loss 113.469
# step 9756500    Train loss  2.6211      Valid loss 113.469

# ## Need to do the following to get the validation data
# ##### Because the run was done on the cluster.

# In[19]:

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    valid_out_data = valid_output_units.eval()


# ## Displaying the reconstruction of first 100 input images in validation by trained sparse autoencoder

# In[8]:

if re.search("ipykernel", sys.argv[0]) :
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    print("Matplotlib is inline")
    get_ipython().magic(u'matplotlib inline')


# In[20]:

fig = plt.figure(figsize=(10,10))
for i in range(100):
    ax = fig.add_subplot(10, 10, i)
    ax.imshow(valid_out_data[i].reshape(image_size, image_size), cmap=cm.gray)


# ### Displaying the first 100 features used to do the reconstruction

# In[10]:

image_size = 28
print(feature_weights.shape)
features=feature_weights.transpose()  
print(features.shape)

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    ax = fig.add_subplot(10, 10, i)
    ax.imshow(features[i].reshape(image_size, image_size), cmap=cm.gray)


# ## Training Softmax Classifier

# In[19]:

test_dataset, testing_labels = testing_data.next_batch(testing_data.num_examples)


# In[23]:

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
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(
      tf.nn.sigmoid(tf.matmul(tf_valid_dataset, weights_hidden1) + biases_hidden1), logit_weights) + logit_biases)
    test_prediction = tf.nn.softmax(tf.matmul(
      tf.nn.sigmoid(tf.matmul(tf_test_dataset, weights_hidden1) + biases_hidden1), logit_weights) + logit_biases)


# In[24]:

def save_for_supervised():
    return save_data(weights, biases, 'out_weights_', 'out_biases_', step)

step = 0
valid_labels = reformat(validation_labels)
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



