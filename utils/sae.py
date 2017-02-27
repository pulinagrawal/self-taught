import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)

class Autoencoder(object):
    """ Sparse Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses encoders and decoders realized by multi-layer perceptrons.

    """
    def __init__(self, network_architecture, learning_rate=0.001, batch_size=100, sparse=False, sparsity=0.1):
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sparse = sparse
        self.rho = sparsity
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture[0]])
        
        # Create autoencoder network
        self.layers = []
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    
    def _create_network(self):
        # Initialize autoencode network weights and biases
        self.network_weights = self._create_weights(*self.network_architecture)
        self.network_baises = self._create_baises(*self.network_architecture)
        self._recognition_network(self.network_weights, self.network_baises)

    def _create_weights(self, *args):
        all_weights = list()
        prev_units = args[0]
        for units in args:
            all_weights.append(tf.Variable(tf.truncated_normal([prev_units, units], stddev=0.01)))
        return all_weights
   
    def _create_baises(self, *args):
        all_baises = list()
        for units in args:
            all_baises.append(tf.Variable(tf.zeros(units)))
        return all_baises
            
    def _recognition_network(self):
        # Creates the encoding part of network. Output is encoder output.
        
        prev_layer_output = self.x
        for layer in [ dict(zip(['weights','baises'], _layer)) for _layer in zip(self.network_weights, self.network_baises)]:
            self.layers.append(self.transfer_fct(layer['weights']*prev_layer_output+layer['baises']))
        self.encoding_layer = self.layers[-1]

    def _generator_network(self):
        # Creates the decoding part of network. Output is reconstruction.
        prev_layer_output = self.layers[-1]
        for layer in [ dict(zip(['weights','baises'], _layer)) for _layer in zip(generator_weights, generator_baises)]:
            self.layers.append(self.transfer_fct(layer['weights']*prev_layer_output+layer['baises']))

    def _KL_divergence(self, units):
        rho = tf.constant(rho)
        int_rho = tf.reduce_sum(units, 0)
        rho_hat = tf.div(int_rho, self.batch_size)
        rho_hat_mean = tf.reduce_mean(rho_hat)
        rho_inv = tf.constant(1.)-rho
        rho_hat_inv = tf.constant(1.)-rho_hat
        klterm = (rho*tf.log(rho/rho_hat))+(rho_in*tf.log(rho_in/rho_hat_in))
        kl_div = tf.reduce_sum(klterm)
        return kl_div
            
    def _create_loss_optimizer(self, reconstruction):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        reconstr_loss = tf.div(tf.nn.l2_loss(tf.sub(reconstruction, self.x)), tf.constant(float(batch_size)))
        
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        for layer in self.layers:
            if layer is not self.layers[-1]: # Reconstruction layer should not be sparse
                latent_loss += self._KL_divergence(layer)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X})
        return cost
    
    def encoding(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.encoding_layer, feed_dict={self.x: X})
    
    def reconstruct(self, X):
        """ Use SAE to reconstruct given data. """
        return self.sess.run(self.layers[-1], feed_dict={self.x: X})