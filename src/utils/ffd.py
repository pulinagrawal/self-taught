
import pickle
import tensorflow as tf

tf.set_random_seed(0)


class FeedForwardNetwork(object):
    """ Autoencoder (AE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses encoders and decoders realized by multi-layer perceptrons.
    Also capable of applying sparse autoencoder with a boolean parameter.

    """

    def __init__(self, network_architecture, session=tf.Session(), learning_rate=0.001,
                 transfer_fct=tf.nn.sigmoid):
        """Initializes a Autoencoder network with network architecture provided in the form of list of
        hidden units from the input layer to the encoding layer. """
        self._network_architecture = network_architecture
        self._starting_learning_rate = learning_rate
        self._sess = session
        self._transfer_fct = transfer_fct
        self._new = True
        self.weights = None
        self.biases = None

        # tf Graph input
        self._global_step = tf.Variable(0, trainable=False)
        self._x = tf.placeholder(tf.float32, [None, network_architecture[0]])
        self._y = tf.placeholder(tf.float32, [None, network_architecture[-1]])
        print('batch size', 'x',  network_architecture[0])

        # Create autoencoder network
        self._layers = []
        self._create_network()
        # corresponding optimizer
        self._create_loss_optimizer(self._layers[-1])

        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

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
        for units in args[1:]:
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
        return all_biases

    def _create_network(self):
        # Initialize autoencoder network weights and biases
        self._network_weights = self.__class__._create_weights(*self._network_architecture)
        self._network_biases = self.__class__._create_biases(*self._network_architecture)
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
            if i == len(self._network_architecture)-2:
                self._encoding_layer = current_layer
            print(prev_layer_output, 'x', layer['weights'], '+', layer['biases'])
            prev_layer_output = current_layer

    def _create_loss_optimizer(self, output):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss
        loss = tf.nn.l2_loss((output-self._y))

        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        #     between the desired sparsity and current sparsity in the latent representation
        #     in all hidden layers

        self.cost = tf.reduce_mean(loss)   # average over batch
        # Use ADAM optimizer
        # TODO Make learning rate dynamic
        self._learning_rate = tf.train.exponential_decay(self._starting_learning_rate, self._global_step, 100, 0.96)
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self.cost, global_step=self._global_step)

    def setup(self):
        """Setup a pre-created network with loaded weights and biases"""
        set_weights = list()
        set_biases = list()
        weights_zip = zip(self._network_weights[:len(self._network_weights)], self.weights)
        for weight_tensor, weight in weights_zip:
            set_weights.append(tf.assign(weight_tensor, weight))
        for biases_tensor, biases in zip(self._network_biases, self.biases):
            set_biases.append(tf.assign(biases_tensor, biases))
        self._sess.run([set_weights, set_biases])

    def partial_fit(self, input_batch, output_labels):
        """Train model based on mini-batch of input data.
        Return cost of mini-batch.
        """
        # TODO Check mini-batch size and input shape

        opt, cost, self.learning_rate, self.weights, self.biases = self._sess.run((self.optimizer, self.cost,
                                                                                   self._learning_rate,
                                                                                   self._network_weights,
                                                                                   self._network_biases),
                                                                                  feed_dict={self._x: input_batch,
                                                                                             self._y: output_labels})
        return cost

    def encoding(self, input_batch):
        """Transform data by mapping it into the latent space."""
        return self._sess.run(self._encoding_layer, feed_dict={self._x: input_batch})

    def loss_on(self, input_batch, labels):
        cost = self._sess.run(self.cost, feed_dict={self._x: input_batch, self._y: labels})
        return cost

    def get_save_state(self):
        save_list = [{'network_architecture': self._network_architecture,
                      'learning_rate': self.learning_rate,
                      'transfer_fct': self._transfer_fct},
                     self.weights[:len(self.weights)],
                     self.biases]
        return save_list

    def save(self, filename, save_state='current'):
        if save_state == 'current':
            save_list = [{'network_architecture': self._network_architecture,
                          'learning_rate': self.learning_rate,
                          'transfer_fct': self._transfer_fct},
                         self.weights[:len(self.weights)],
                         self.biases]
        else:
            save_list = save_state

        with open(filename, 'wb') as save_file:
            pickle.dump(save_list, save_file)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as load_file:
            load_list = pickle.load(load_file)
            instance = cls(**load_list[0])
            instance.weights = load_list[1]
            instance.biases = load_list[2]
            instance.setup()
        return instance
