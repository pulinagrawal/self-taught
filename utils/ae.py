
import pickle
import tensorflow as tf

from functools import wraps

tf.set_random_seed(0)
import numpy as np
inp = np.random.random_sample([100,784])


class Autoencoder(object):
    """ Autoencoder (AE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses encoders and decoders realized by multi-layer perceptrons.
    Also capable of applying sparse autoencoder with a boolean parameter.

    """
    def __init__(self, _network_architecture, session=tf.Session(), learning_rate=0.001,
                 batch_size=100, sparse=False, sparsity=0.1, transfer_fct=tf.nn.sigmoid, tied_weights=True):
        self._network_architecture = _network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sparse = sparse
        self.rho = sparsity
        self._sess = session
        self._transfer_fct = transfer_fct
        self._tied_weights = tied_weights

        # tf Graph input
        self._x = tf.placeholder(tf.float32, [self.batch_size, _network_architecture[0]])
        print(self.batch_size, 'x',  _network_architecture[0])

        # Create autoencoder network
        self._layers = []
        self._create_network()
        # corresponding optimizer
        self._create_loss_optimizer(self._layers[-1])

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self._sess.run(init)

    @classmethod
    def _create_tied_weights(cls, *args):
        all_weights = list()
        forward_weights = list()
        prev_units = args[0]
        for units in args[1:]:
            weights = tf.Variable(tf.truncated_normal([prev_units, units], stddev=0.01))
            all_weights.append(weights)
            forward_weights.append(weights)
            print(weights.get_shape())
            prev_units = units
        for forward_weight in reversed(forward_weights):
            backward_weight = tf.transpose(forward_weight)
            all_weights.append(backward_weight)
            print(backward_weight.get_shape())
        return all_weights

    @classmethod
    def _create_weights(cls, *args):
        all_weights = list()
        prev_units = args[0]
        for units in args[1:]+list(reversed(args[:-1])):
            weights = tf.Variable(tf.truncated_normal([prev_units, units], stddev=0.01))
            all_weights.append(weights)
            print(weights.get_shape())
            prev_units = units
        return all_weights

    @classmethod
    def _create_biases(cls, *args):
        all_biases = list()
        for units in args[1:]:
            all_biases.append(tf.Variable(tf.zeros(units)))
            print(units)
        for units in reversed(args[:-1]):
            all_biases.append(tf.Variable(tf.zeros(units)))
            print(units)
        return all_biases

    def _create_network(self):
        # Initialize autoencoder network weights and biases
        if self._tied_weights:
            self._network_weights = Autoencoder._create_tied_weights(*self._network_architecture)
        else:
            self._network_weights = Autoencoder._create_weights(*self._network_architecture)
        self._network_biases = Autoencoder._create_biases(*self._network_architecture)
        self._hook_em_up()

    def _hook_em_up(self):
        # Creates the encoding part of network. Output is encoder output.
        print("Network")
        prev_layer_output = self._x
        layers = [dict(zip(['weights', 'biases'], _layer))
                      for _layer in zip(self._network_weights, self._network_biases)]
        for i, layer in enumerate(layers):
            current_layer = self._transfer_fct(tf.matmul(prev_layer_output, layer['weights'])+layer['biases'])
            self._layers.append(current_layer)
            if i == len(self._network_architecture)-1:
                self._encoding_layer = current_layer
            print(prev_layer_output,'x',layer['weights'],'+',layer['biases'])
            prev_layer_output = current_layer

    def _KL_divergence(self, units):
        rho = tf.constant(self.rho)
        int_rho = tf.reduce_sum(units, 0)
        rho_hat = tf.div(int_rho, self.batch_size)
        rho_inv = tf.constant(1.)-rho
        rho_hat_inv = tf.constant(1.)-rho_hat
        kl_term = (rho*tf.log(rho/rho_hat))+(rho_inv*tf.log(rho_inv/rho_hat_inv))
        kl_div = tf.reduce_sum(kl_term)
        return kl_div

    def _create_loss_optimizer(self, reconstruction):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss
        reconstruction_loss = tf.nn.l2_loss((reconstruction-self._x))/tf.constant(float(self.batch_size))

        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        #     between the desired sparsity and current sparsity in the latent representation
        #     in all hidden layers
        latent_loss = tf.constant(0.)

        if self.sparse:
            for layer in self._layers:
                if layer is not self._layers[-1]:  # Reconstruction layer should not be sparse
                    latent_loss += self._KL_divergence(layer)

        self.cost = tf.reduce_mean(reconstruction_loss + latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def sessioned(session_func):
        @wraps(session_func)
        def sessioned_func(*args, **kwargs):
            global self
            if self._sess is None:
                self._sess = tf.Session()
                if self.weights is not None:
                    set_weights = tf.assign(self._network_weights, self.weights)
                    set_biases = tf.assign(self._network_biases, self.biases)
                    self._sess.run([set_weights, set_biases])
                else:
                    self._sess.run(tf.global_variables_initializer())
            return session_func(*args, **kwargs)
        return sessioned_func

    @sessioned
    def partial_fit(self, input_batch):
        """Train model based on mini-batch of input data.
        Return cost of mini-batch.
        """
        # TODO Check mini-batch size and input shape
        opt, cost, self.weights, self.biases = self._sess.run((self.optimizer, self.cost,
                                                               self._network_weights, self._network_biases),
                                                              feed_dict={self._x: input_batch})
        return cost

    @sessioned
    def encoding(self, input_tensor):
        """Transform data by mapping it into the latent space."""
        return self._sess.run(self._encoding_layer, feed_dict={self._x: input_tensor})

    @sessioned
    def reconstruct(self, input_tensor):
        """ Use Autoencoder to reconstruct given data. """
        return self._sess.run(self._layers[-1], feed_dict={self._x: input_tensor})

    def save(self, filename):
        with open(filename, 'wb') as save_file:
            pickle.dump(self, save_file)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as load_file:
            instance = pickle.load(load_file)
        return instance