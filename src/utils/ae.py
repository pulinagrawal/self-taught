
import pickle
import tensorflow as tf

tf.set_random_seed(0)


class Autoencoder(object):
    """ Autoencoder (AE) implemented using TensorFlow.
    
    This implementation uses encoders and decoders realized by multi-layer perceptrons.
    Also capable of applying sparse autoencoder with a boolean parameter.

    """
    def __init__(self, network_architecture, name='ae', learning_rate=0.00001,
                 sparse=False, sparsity=0.1, transfer_fct=tf.nn.sigmoid, tied_weights=True):
        """Initializes a Autoencoder network with network architecture provided in the form of list of
        hidden units from the input layer to the encoding layer. """
        self._network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.sparse = sparse
        self.rho = sparsity
        self._transfer_fct = transfer_fct
        self._tied_weights = tied_weights
        self._new = True
        self.beta = 3
        self.weights = None
        self.biases = None
        self._name = name
        self.step = 0

        self.graph = tf.Graph()
        with self.graph.as_default():
            #TODO: Serialization maybe possible if the session object is instantiated later.
            # tf Graph input
            with tf.name_scope(self._name+'/input'):
                self._x = tf.placeholder(tf.float32, [None, network_architecture[0]], name='input')
            print('batch size', 'x',  network_architecture[0])

            # Create autoencoder network
            self._layers = []
            self._create_network()

            for i, weights in enumerate(self._network_weights):
                if i >= len(self._network_weights)/2:

                    for j in range(i+1, len(self._network_weights)):
                        weights = tf.matmul(weights, self._network_weights[j])

                    shape = weights.get_shape()
                    shape = [shape.dims[0].value, shape.dims[1].value]
                    if (shape[1] ** 0.5) - int(shape[1] ** 0.5) == 0:
                        image = tf.reshape(weights, [shape[0], int(shape[1] ** 0.5), int(shape[1] ** 0.5), 1])
                    else:
                        image = tf.reshape(weights, [shape[0], shape[1], 1, 1])

                    tf.summary.image('hidden{0}_images'.format(len(self._network_weights)-i), image)

            # corresponding optimizer
            self._create_loss_optimizer(self._layers[-1])

            self.summary = tf.summary.merge_all()

        self.start_session()


    def start_session(self):
        self._sess = tf.Session(graph=self.graph)

        self.train_writer = tf.summary.FileWriter('train_summary', self._sess.graph)
        self.test_writer = tf.summary.FileWriter('test_summary')

        # Initializing the tensor flow variables
        with self.graph.as_default():
            init = tf.initialize_all_variables()

        # Launch the session
        self._sess.run(init)

    def stop_session(self):
        self._sess.close()

    def _create_tied_weights(self, *args):
        all_weights = list()
        forward_weights = list()
        prev_units = args[0]
        layer_num = 0
        for units in args[1:]:
            with tf.name_scope(self._name+'/hidden{0}/'.format(layer_num)):
                weights = tf.Variable(tf.truncated_normal([prev_units, units], stddev=0.01), name='weights')
            all_weights.append(weights)
            forward_weights.append(weights)
            print(weights.get_shape())
            prev_units = units
            layer_num += 1
        for forward_weight in reversed(forward_weights):
            with tf.name_scope(self._name+'/hidden{0}/'.format(layer_num)):
                backward_weight = tf.transpose(forward_weight, name='weights')
            all_weights.append(backward_weight)
            print(backward_weight.get_shape())
            layer_num += 1
        return all_weights

    def _create_weights(self, *args):
        all_weights = list()
        prev_units = args[0]
        for layer_num, units in enumerate(args[1:]+list(reversed(args[:-1]))):
            with tf.name_scope(self._name+'/hidden{0}/'.format(layer_num)):
                weights = tf.Variable(tf.truncated_normal([prev_units, units], stddev=0.01), name='weights')
            all_weights.append(weights)
            print(weights.get_shape())
            prev_units = units
        return all_weights

    def _create_biases(self, *args):
        all_biases = list()
        layer_num = 0
        for units in args[1:]:
            with tf.name_scope(self._name+'/hidden{0}/'.format(layer_num)):
                all_biases.append(tf.Variable(tf.zeros([units]), name='biases'))
            print(units)
            layer_num += 1
        for units in reversed(args[:-1]):
            with tf.name_scope(self._name+'/hidden{0}/'.format(layer_num)):
                all_biases.append(tf.Variable(tf.zeros([units]), name='biases'))
            print(units)
            layer_num += 1
        return all_biases

    def _create_network(self):
        # Initialize autoencoder network weights and biases
        if self._tied_weights:
            self._network_weights = self._create_tied_weights(*self._network_architecture)
        else:
            self._network_weights = self._create_weights(*self._network_architecture)
        self._network_biases = self._create_biases(*self._network_architecture)
        self._hook_em_up()

    def _hook_em_up(self):
        # Creates the encoding part of network. Output is encoder output.
        print("Network")
        prev_layer_output = self._x
        layers = [dict(zip(['weights', 'biases'], _layer))
                  for _layer in zip(self._network_weights, self._network_biases)]
        for i, layer in enumerate(layers):
            with tf.name_scope(self._name+'/hidden{0}/'.format(i)):
                if i == len(self._network_architecture)-1:
                    current_layer = self._transfer_fct((tf.matmul(prev_layer_output, layer['weights'])+layer['biases']), name='units')
                else:
                    current_layer = self._transfer_fct(tf.multiply((tf.matmul(prev_layer_output, layer['weights'])+layer['biases']),8), name='units')
            self._layers.append(current_layer)
            if i == len(self._network_architecture)-2:
                print("i = ", i, "encoding layer = ", current_layer)
                self._encoding_layer = current_layer
            print(prev_layer_output, 'x', layer['weights'], '+', layer['biases'])
            prev_layer_output = current_layer

    def _KL_divergence(self, units):
        with tf.name_scope(self._name+'/sparse_regularization/'):
            rho = tf.constant(self.rho, name='rho')
            rho_hat = units
            rho_inv = tf.constant(1.)-rho
            #rho_inv = tf.Print(rho_inv, [rho_inv, tf.shape(rho_inv), 'rho_inv'])
            rho_hat_inv = tf.constant(1.)-rho_hat
            #rho_hat_inv = tf.Print(rho_hat_inv, [rho_hat_inv, tf.shape(rho_hat_inv), 'rho_hat_inv'])
            term1 = rho*tf.log(rho/rho_hat)
            #term1 = tf.Print(term1, [term1, tf.shape(term1), 'term1'])
            term2 = rho_inv*tf.log(rho_inv/rho_hat_inv)
            #term2 = tf.Print(term2 , [term2 , tf.shape(term2 ), 'term2 '])
            kl_term = term1 + term2
            #kl_term = tf.Print(kl_term, [kl_term, tf.shape(kl_term), 'kl_term'])
            kl_div = tf.reduce_sum(kl_term, 1, name='kl_div')
            #kl_div = tf.Print(kl_div , [kl_div , tf.shape(kl_div ), 'kl_div '])
            avg_kl_div = tf.reduce_mean(kl_div)
        return avg_kl_div

    def _create_loss_optimizer(self, reconstruction_tensor):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss
        with tf.name_scope(self._name+'/loss/'):
            self.reconstruction_loss = tf.reduce_mean(tf.nn.l2_loss(reconstruction_tensor-self._x), name='reconstruction_loss')

            # 2.) The latent loss, which is defined as the Kullback Leibler divergence
            #     between the desired sparsity and current sparsity in the latent representation
            #     in all hidden layers
            latent_loss = tf.constant(0.)

            if self.sparse:
                for i, layer in enumerate(self._layers):
                    if layer is not self._layers[-1]:  # Reconstruction layer should not be sparse
                        latent_loss = tf.add(latent_loss, self._KL_divergence(layer), name='layer{0}_latent_loss'.format(i))

            self.latent_loss = tf.add(latent_loss, 0, name='latent_loss')

            #self.reconstruction_loss = tf.Print(self.reconstruction_loss, [self.reconstruction_loss, 'reconstruction_loss'])
            #self.latent_loss = tf.Print(self.latent_loss , [self.latent_loss , 'latent_loss '])

            self.cost = tf.add(self.reconstruction_loss, self.beta*self.latent_loss, name='cost')   # average over batch
            tf.summary.scalar('cost', self.cost)
            tf.summary.scalar('latent_loss', self.latent_loss)
            # Use ADAM optimizer
            # TODO Make learning rate dynamic
            self.optimizer = \
                tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost, name='optimizer')

    def setup(self):
        """Setup a pre-created network with loaded weights and biases"""
        set_weights = list()
        set_biases = list()
        weights_zip = zip(self._network_weights[:len(self._network_weights)//2], self.weights)
        for weight_tensor, weight in weights_zip:
            set_weights.append(tf.assign(weight_tensor, weight))
        for biases_tensor, biases in zip(self._network_biases, self.biases):
            set_biases.append(tf.assign(biases_tensor, biases))
        self._sess.run([set_weights, set_biases])

    def partial_fit(self, input_batch):
        """Train model based on mini-batch of input data.
        Return cost of mini-batch.
        """
        # TODO Check mini-batch size and input shape
        summary, opt, cost, self.weights, self.biases = self._sess.run((self.summary, self.optimizer, self.cost,
                                                                        self._network_weights, self._network_biases),
                                                                       feed_dict={self._x: input_batch})
        self.train_writer.add_summary(summary, self.step)
        self.step += 1
        return cost

    def encoding(self, input_tensor):
        """Transform data by mapping it into the latent space."""
        return self._sess.run(self._encoding_layer, feed_dict={self._x: input_tensor})

    def loss(self, input_tensor):
        return self._sess.run((self.cost, self.reconstruction_loss), feed_dict={self._x: input_tensor})

    def reconstruction_loss(self, input_tensor):
        loss = self._sess.run((self.reconstruction_loss,), feed_dict={self._x: input_tensor})
        return loss

    def reconstruct(self, input_tensor):
        """ Use Autoencoder to reconstruct given data. """
        return self._sess.run(self._layers[-1], feed_dict={self._x: input_tensor})

    def get_save_state(self):
        save_list = [{'network_architecture': self._network_architecture,
                      'learning_rate': self.learning_rate,
                      'sparse': self.sparse,
                      'sparsity': self.rho,
                      'transfer_fct': self._transfer_fct,
                      'tied_weights': self._tied_weights},
                     self.weights[:len(self.weights)//2],
                     self.biases]
        return save_list

    def save(self, filename, save_state='current'):
        if save_state == 'current':
            save_list = [{'network_architecture': self._network_architecture,
                          'learning_rate': self.learning_rate,
                          'sparse': self.sparse,
                          'sparsity': self.rho,
                          'transfer_fct': self._transfer_fct,
                          'tied_weights': self._tied_weights},
                         self.weights[:len(self.weights)//2],
                         self.biases]
        else:
            save_list = save_state

        with open(filename, 'wb') as save_file:
            pickle.dump(save_list, save_file)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as load_file:
            load_list = pickle.load(load_file, encoding='latin1')
            instance = cls(**load_list[0])
            instance.weights = load_list[1]
            instance.biases = load_list[2]
            instance.setup()
        return instance

    @classmethod
    def save_pickle(cls, object, filename):
        with open(filename, 'wb') as file:
            pickle.dump(object, file)

    @classmethod
    def load_pickle(cls, filename):
        with open(filename, 'rb') as file:
            autoenc = pickle.load(file)

        autoenc.start_session()
        autoenc.setup()

        return autoenc
