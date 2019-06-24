import pickle
import numpy as np
import tensorflow as tf


class Autoencoder(object):
    """ Autoencoder (AE) implemented using TensorFlow.
    
    This implementation uses encoders and decoders realized by multi-layer perceptrons.
    Also capable of applying sparse autoencoder with a boolean parameter.

    """

    def __init__(self, network_architecture, name='ae', learning_rate=0.001,
                 sparse=True, sparsity=0.1, transfer_fct=tf.nn.sigmoid, beta=3, step=0,
                 reconstruction_batch_size=100, lambda_=0, tied_weights=True,
                 keep_prob=0.5, denoise_keep_prob=0.9, dynamic_learning_rate=False,
                 momentum=0.8, tf_multiplier=10, zero_noise=True, logdir='summary', summary_image=False):
        """Initializes a Autoencoder network with network architecture provided in the form of list of
        hidden units from the input layer to the encoding layer. """
        self._network_architecture = network_architecture
        self._starting_learning_rate = learning_rate
        self.dynamic_learning_rate = dynamic_learning_rate
        self.sparse = sparse
        self.rho = sparsity
        self._transfer_fct = transfer_fct
        self.zero_noise = zero_noise
        self._tied_weights = tied_weights
        self._new = True
        self.weights = None
        self.biases = None
        self._name = name
        self.step = step
        self.lambda_ = lambda_
        self.beta = beta
        self.logdir = logdir
        self.momentum = momentum
        self.multiplier = tf_multiplier
        self.keep_prob = keep_prob
        self.denoise_keep_prob = denoise_keep_prob
        self._reconstruction_batch_size = reconstruction_batch_size

        self.graph = tf.Graph()
        self.summaries = {}
        self.summary_image=summary_image
        with self.graph.as_default():
            # TODO: Serialization maybe possible if the session object is instantiated later.
            # tf Graph input
            self._global_step = tf.Variable(0, trainable=False)
            with tf.name_scope(self._name + '/input'):
                self._x = tf.placeholder(tf.float32, [None, network_architecture[0]], name='input')
            print('batch size', 'x', network_architecture[0])
            self._keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self._denoise_keep_prob = tf.placeholder(tf.float32, name='denoise_keep_prob')
            recon_batch_size = tf.constant(self._reconstruction_batch_size)

            # Create autoencoder network
            self._layers = []
            self._create_network()

            # corresponding optimizer
            self._create_loss_optimizer(self._layers[-1])

            if self.sparse:
                self.summary = tf.summary.merge(
                    [self.summaries['cost'], self.summaries['latent_loss'], self.summaries['beta'],
                     self.summaries['learning_rate'], self.summaries['avg_rho_hat']])
            else:
                self.summary = tf.summary.merge(
                    [self.summaries['cost'], self.summaries['latent_loss'], self.summaries['beta'],
                     self.summaries['learning_rate']])

            if self.summary_image:
                with tf.name_scope(self._name + '/reconstruction/'):
                    output_image = self.reshape_tensor_for_display(self._layers[-1], recon_batch_size)
                    input_image = self.reshape_tensor_for_display(self._x, recon_batch_size)
                    self.summaries['reconstructed_image'] = tf.summary.image('reconstructed_images', output_image)
                    self.summaries['input_image'] = tf.summary.image('input_images', input_image)
                    self.summary_feature_images()
                    self.image_summaries = tf.summary.merge([self.summaries['reconstructed_image'],
                                                             self.summaries['input_image'],
                                                             self.summaries['feature_images']
                                                             ])
        self.start_session()

    def reshape_tensor_for_display(self, tensor, batch_size='default'):
        shape = tensor.get_shape()
        shape = [shape.dims[0].value, shape.dims[1].value]
        if batch_size == 'default':
            batch_size = shape[0]
        if (shape[1] ** 0.5) - int(shape[1] ** 0.5) == 0:
            image = tf.reshape(tensor, [batch_size, int(shape[1] ** 0.5), int(shape[1] ** 0.5), 1])
        else:
            image = tf.reshape(tensor, [batch_size, shape[1], 1, 1])

        return image

    def summary_feature_images(self, count=3):
        for i, weights in enumerate(self._network_weights):
            if i >= len(self._network_weights) / 2:

                for j in range(i + 1, len(self._network_weights)):
                    weights = tf.matmul(weights, self._network_weights[j])
                    weights.eval()
                image = self.reshape_tensor_for_display(weights)

                self.summaries['feature_images'] = tf.summary.image(
                    'hidden{0}_images'.format(len(self._network_weights) - i), image[:count])

    def start_session(self):
        self._sess = tf.Session(graph=self.graph)

        self.train_writer = tf.summary.FileWriter(self.logdir, self._sess.graph)

        # Initializing the tensor flow variables
        with self.graph.as_default():
            init = tf.global_variables_initializer()

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
            with tf.name_scope(self._name + '/hidden{0}/'.format(layer_num)):
                weights = tf.Variable(tf.truncated_normal([prev_units, units], stddev=0.01), name='weights')
            all_weights.append(weights)
            forward_weights.append(weights)
            print(weights.get_shape())
            prev_units = units
            layer_num += 1
        for forward_weight in reversed(forward_weights):
            with tf.name_scope(self._name + '/hidden{0}/'.format(layer_num)):
                backward_weight = tf.transpose(forward_weight, name='weights')
            all_weights.append(backward_weight)
            print(backward_weight.get_shape())
            layer_num += 1
        return all_weights

    def _create_weights(self, *args):
        all_weights = list()
        prev_units = args[0]
        print(list(args[1:]))
        print(list(reversed(args[:-1])))
        for layer_num, units in enumerate(list(args[1:]) + list(reversed(args[:-1]))):
            with tf.name_scope(self._name + '/hidden{0}/'.format(layer_num)):
                weights = tf.Variable(tf.truncated_normal([prev_units, units], stddev=0.01)/tf.sqrt(prev_units), name='weights')
            all_weights.append(weights)
            print(weights.get_shape())
            prev_units = units
        return all_weights

    def _create_biases(self, *args):
        all_biases = list()
        layer_num = 0
        for units in args[1:]:
            with tf.name_scope(self._name + '/hidden{0}/'.format(layer_num)):
                all_biases.append(tf.Variable(tf.zeros([units]), name='biases'))
            print(units)
            layer_num += 1
        for units in reversed(args[:-1]):
            with tf.name_scope(self._name + '/hidden{0}/'.format(layer_num)):
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
        zero = False
        prev_layer_output = self._x
        # TODO May implement denoising
        # prev_layer_output = tf.nn.dropout(prev_layer_output, self._denoise_keep_prob)

        if not self.zero_noise:
            prev_layer_output = prev_layer_output + tf.random_normal(shape=tf.shape(self._x), mean=0.0,
                                                                 stddev=1-self._denoise_keep_prob)
        else:
            noise = tf.random_uniform(tf.shape(prev_layer_output), maxval=1)
            prev_layer_output = tf.where(noise < self._denoise_keep_prob, prev_layer_output, tf.zeros(tf.shape(prev_layer_output), dtype=tf.float32))

        # prev_layer_output = tf.Print(prev_layer_output, [prev_layer_output, tf.shape(prev_layer_output), 'input'])
        layers = [dict(zip(['weights', 'biases'], _layer))
                  for _layer in zip(self._network_weights, self._network_biases)]
        for i, layer in enumerate(layers):
            with tf.name_scope(self._name + '/hidden{0}/'.format(i)):
                current_layer = self._transfer_fct((tf.matmul(prev_layer_output, layer['weights'])+layer['biases'])*self.multiplier, name='units')
                #current_layer = tf.Print(current_layer, [current_layer, tf.shape(current_layer), 'current_layer'])
            self._layers.append(current_layer)
            if i == len(self._network_architecture) - 2:
                print("i = ", i, "encoding layer = ", current_layer)
                self._encoding_layer = current_layer
                current_layer = tf.nn.dropout(current_layer, self._keep_prob)
            print(prev_layer_output, 'x', layer['weights'], '+', layer['biases'])
            prev_layer_output = current_layer

    def _KL_divergence(self, units):
        with tf.name_scope(self._name + '/sparse_regularization/'):
            rho = tf.constant(self.rho, name='rho')
            rho_hat = tf.reduce_mean(units, 0)
            # rho_hat = tf.Print(rho_hat, [rho_hat, tf.shape(rho_hat), 'rho_hat'])
            self.summaries['avg_rho_hat'] = tf.summary.scalar('avg_rho_hat', tf.reduce_mean(rho_hat))
            rho_inv = tf.constant(1.) - rho
            # rho_inv = tf.Print(rho_inv, [rho_inv, tf.shape(rho_inv), 'rho_inv'])
            rho_hat_inv = tf.constant(1.) - rho_hat
            # rho_hat_inv = tf.Print(rho_hat_inv, [rho_hat_inv, tf.shape(rho_hat_inv), 'rho_hat_inv'])
            term1 = rho * tf.log(rho / rho_hat)
            # term1 = tf.Print(term1, [term1, tf.shape(term1), 'term1'])
            term2 = rho_inv * tf.log(rho_inv / rho_hat_inv)
            # term2 = tf.Print(term2 , [term2 , tf.shape(term2 ), 'term2 '])
            kl_term = term1 + term2
            # kl_term = tf.Print(kl_term, [kl_term, tf.shape(kl_term), 'kl_term'])
            kl_div = tf.reduce_sum(kl_term, 0, name='kl_div')
            # kl_div = tf.Print(kl_div , [kl_div , tf.shape(kl_div ), 'kl_div '])
        return kl_div

    def _create_loss_optimizer(self, reconstruction_tensor):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss
        with tf.name_scope(self._name + '/loss/'):
            loss_sub = tf.subtract(reconstruction_tensor, self._x)
            # loss_sub = tf.Print(loss_sub , [loss_sub , tf.shape(loss_sub ), 'loss_sub'])
            l2_loss = tf.reduce_sum(tf.square(loss_sub), 1)
            # l2_loss = tf.Print(l2_loss, [l2_loss, tf.shape(l2_loss)], 'l2_loss')
            self.reconstruction_loss = tf.reduce_mean(l2_loss, name='reconstruction_loss')
            # self.reconstruction_loss = tf.nn.l2_loss(loss_sub, name='reconstruction_loss')
            # self.reconstruction_loss = tf.Print(self.reconstruction_loss, [self.reconstruction_loss, 'self.reconstruction_loss '])

            # 2.) The latent loss, which is defined as the Kullback Leibler divergence
            #     between the desired sparsity and current sparsity in the latent representation
            #     in all hidden layers
            latent_loss = tf.constant(0.)

            if self.sparse:
                for i, layer in enumerate(self._layers):
                    if layer is not self._layers[-1]:  # Reconstruction layer should not be sparse
                        latent_loss = tf.add(latent_loss, self._KL_divergence(layer),
                                             name='layer{0}_latent_loss'.format(i))

            self.latent_loss = tf.add(latent_loss, 0, name='latent_loss')

            # self.latent_loss = tf.Print(self.latent_loss , [self.latent_loss , 'latent_loss '])

            weight_loss = tf.constant(0.)

            for i, weights in enumerate(self._network_weights):
                if weights is not self._network_weights[-1]:
                    weight_loss = weight_loss + tf.reduce_sum(tf.square(weights))

            self.weight_loss = tf.add(weight_loss, 0, name='weight_loss')
            # self.weight_loss = tf.Print(self.weight_loss, [self.weight_loss , 'weight_loss '])

            self.cost = tf.add(tf.add(self.reconstruction_loss, (self.lambda_/2)*self.weight_loss), self.beta*self.latent_loss, name='cost')   # average over batch
            # self.cost = self.reconstruction_loss
            self.summaries['cost'] = tf.summary.scalar('cost', self.cost)
            self.summaries['latent_loss'] = tf.summary.scalar('latent_loss', self.latent_loss)
            # Use ADAM optimizer
            # TODO Make learning rate dynamic
            self.summaries['beta'] = tf.summary.scalar('beta', self.beta)
            if self.dynamic_learning_rate:
                self._learning_rate = tf.train.exponential_decay(self._starting_learning_rate, self._global_step, 500,
                                                                 0.96)
            else:
                self._learning_rate = self._starting_learning_rate
            self.summaries['learning_rate'] = tf.summary.scalar('learning_rate', self._learning_rate)
            self.optimizer = \
                tf.train.RMSPropOptimizer(learning_rate=self._learning_rate,
                                          momentum=self.momentum).minimize(self.cost,
                                                                           global_step=self._global_step,
                                                                           name='optimizer')

    def setup(self):
        """Setup a pre-created network with loaded weights and biases"""
        set_weights = list()
        set_biases = list()
        weights_zip = zip(self._network_weights[:len(self._network_weights) // 2], self.weights)
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
        summary, opt, cost, self.weights, self.biases = self._sess.run((self.summary,
                                                                        self.optimizer, self.cost,
                                                                        self._network_weights, self._network_biases),
                                                                       feed_dict={self._x: input_batch,
                                                                                  self._keep_prob: self.keep_prob,
                                                                                  self._denoise_keep_prob: self.denoise_keep_prob
                                                                                  })
        self.train_writer.add_summary(summary, self.step)
        self.step += 1
        return cost

    def encoding(self, input_tensor):
        """Transform data by mapping it into the latent space."""
        return self._sess.run(self._encoding_layer, feed_dict={self._x: input_tensor,
                                                               self._keep_prob: 1.0,
                                                               self._denoise_keep_prob: 1.0
                                                               })

    def loss(self, input_tensor):
        output = self._sess.run((self.cost, self.reconstruction_loss, self._layers[-1]), feed_dict={self._x: input_tensor,
                                                                                self._keep_prob: 1.0,
                                                                                self._denoise_keep_prob: 1.0
                                                                                })
        print(input_tensor, 'input')
        print(output[2], 'output')
        return (output[0], output[1])
    def reconstruction_loss(self, input_tensor):
        loss = self._sess.run((self.reconstruction_loss,), feed_dict={self._x: input_tensor,
                                                                      self._keep_prob: 1.0,
                                                                      self._denoise_keep_prob: 1.0
                                                                      })
        return loss

    def reconstruct(self, input_tensor):
        """ Use Autoencoder to reconstruct given data. """
        if self.summary_image:
            reconstruction, reconstruction_summary = self._sess.run((self._layers[-1], self.image_summaries),
                                                                    feed_dict={self._x: input_tensor,
                                                                               self._keep_prob: 1.0,
                                                                               self._denoise_keep_prob: 1.0
                                                                               })
            self.train_writer.add_summary(reconstruction_summary, self.step)
        else:
            reconstruction = self._sess.run((self._layers[-1]),
                                            feed_dict={self._x: input_tensor,
                                            self._keep_prob: 1.0,
                                            self._denoise_keep_prob: 1.0
                                            })

        return reconstruction

    def get_save_state(self):
        save_list = [{'network_architecture': self._network_architecture,
                      'learning_rate': self._starting_learning_rate,
                      'sparse': self.sparse,
                      'sparsity': self.rho,
                      'transfer_fct': self._transfer_fct,
                      'lambda_': self.lambda_,
                      'beta': self.beta,
                      'step': self.step,
                      'logdir': self.logdir,
                      'tied_weights': self._tied_weights},
                     self.weights[:len(self.weights) // 2],
                     self.biases]
        return save_list

    def save(self, filename, save_state='current'):
        if save_state == 'current':
            save_list = self.get_save_state()
        else:
            save_list = save_state

        with open(filename, 'wb') as save_file:
            pickle.dump(save_list, save_file)

    @classmethod
    def load_model(cls, filename, logdir='original'):
        with open(filename, 'rb') as load_file:
            load_list = pickle.load(load_file, encoding='latin1')
            if logdir != 'original':
                load_list[0]['logdir'] = logdir
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

    @classmethod
    def get_identity_encoder(cls, input_size, name='identity_ae'):
        feature_network = cls([input_size, input_size], name=name,
                                         sparse=False, transfer_fct=tf.nn.relu, tf_multiplier=1)
        feature_network.weights = [np.identity(input_size)]
        feature_network.biases = [np.zeros(input_size)]
        feature_network.setup()
        return feature_network

    @property
    def encoding_size(self):
        return self._network_architecture[-1]
