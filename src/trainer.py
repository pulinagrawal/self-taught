import numpy as np
import datetime as dt
import os
import csv

from utils import ae
from utils import ffd
import utils.generator as gen_utils
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

tf.logging.set_verbosity(tf.logging.INFO)
LEARNING_RATE_LIMIT = 0.00001
EPOCH_WINDOW_FOR_STOPPING = 50


class SelfTaughtTrainer(object):
    """This class creates a trainer object that will assist in self-taught training of a network.
    The training happens in two step. run_unsupervised_training() trains the network on unlabelled data.
    run_supervised_training() trains the network on labelled data. You need to provide this class with an
    network object."""

    def __init__(self, feature_network, output_network, batch_size,
                 unlabelled, labelled, validation, test, save_filename, run_folder=str(dt.datetime.now()),
                 early_stopping=True, max_epochs=20000):
        self._unlabelled = unlabelled
        print("Unlabelled Examples", self._unlabelled.num_examples)
        self._labelled = labelled
        print("Labelled Examples", self._labelled.num_examples)
        self._validation = validation
        print("Validation Examples", self._validation.num_examples)
        self._test = test
        print("Test Examples", self._test.num_examples)
        self._feature_network = feature_network
        self._output_network = output_network
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._early_stopping = early_stopping
        self._learning_rate_limit = LEARNING_RATE_LIMIT
        self._run_folder = run_folder.replace(' ', '_').replace(':', '-').replace('.', '-')
        os.mkdir(self._run_folder)
        self._save_filename = os.path.join(self._run_folder, save_filename)

    @classmethod
    def from_only_labelled(cls, feature_network, output_network, batch_size,
                           data, save_filename, run_folder=str(dt.datetime.now()), early_stopping=True,
                           unlabelled_pct=80, max_epochs=20000):
        num_unlabelled = int((unlabelled_pct/100.0)*data.train.num_examples)
        unlabelled = data.train.next_batch(num_unlabelled)
        unlabelled_data = DataSet(unlabelled[0], unlabelled[1], reshape=False)
        #TODO Probablu Only for mnist data - Dataset initializer scales data by 1/255
        unlabelled_data._images = unlabelled[0]
        labelled = data.train.next_batch(data.train.num_examples-num_unlabelled)
        labelled_data = DataSet(labelled[0], labelled[1], reshape=False)
        #TODO Probablu Only for mnist data - Dataset initializer scales data by 1/255
        labelled_data._images = labelled[0]
        validation = data.validation
        test = data.test
        return cls(feature_network, output_network, batch_size, unlabelled_data,
                   labelled_data, validation, test, save_filename, run_folder, early_stopping, max_epochs)

    @staticmethod
    @gen_utils.ready_generator
    def list_of_values_stopping_criterion():    #TODO Not general: only accepts list of matrices
        stop_cond = False
        prev_values = yield
        new_values = yield stop_cond
        while True:
            new_values_temp = new_values
            diff_weights = np.subtract(new_values, prev_values)
            pct_change_values = np.absolute(np.divide(diff_weights, np.array(prev_values)))
            max_diff_pct = np.max(np.concatenate([change.flatten() for change in pct_change_values], axis=0))
            stop_cond = max_diff_pct < .00001
            # TODO May need a if breaking (so that generator terminates smoothly?)
            new_values = yield stop_cond
            prev_values = new_values_temp

    @staticmethod
    @gen_utils.ready_generator
    def historic_change_stopping_criterion():
        epoch_window = EPOCH_WINDOW_FOR_STOPPING
        i = 0
        stop_cond = False
        prev_values = {}
        while True:
            prev_values[i % epoch_window] = yield stop_cond
            i = i+1
            if i > epoch_window:
                stop_cond = prev_values[i % epoch_window] > prev_values[(i+1) % epoch_window]

    @staticmethod
    @gen_utils.ready_generator
    def early_stopping_criterion():
        patience_window = EPOCH_WINDOW_FOR_STOPPING
        i = 0
        stop_cond = False
        best_value = float('inf')
        while True:
            new_value = yield stop_cond
            i = i+1
            if i > patience_window:
                stop_cond = True
            if new_value < best_value:
                i = 0
                best_value = new_value

    def run_unsupervised_training(self):
        self.loss_log = [('training_loss', 'validation_loss', 'validation_reconstruction_loss')]
        stop_for_reconstruction_loss = SelfTaughtTrainer.early_stopping_criterion()
        save_dict={}
        last_epoch = 0
        validation_loss = 0
        validation_reconstruction_loss = 0
        fig = plt.figure(figsize=(10, 10))
        start_flag = True
        while self._unlabelled.epochs_completed < self._max_epochs:
            training_loss = self._feature_network.partial_fit(self._unlabelled.next_batch(self._batch_size)[0])
            self.loss_log.append((training_loss, validation_loss, validation_reconstruction_loss))

            if self._unlabelled.epochs_completed > last_epoch or start_flag:
                start_flag = False
                last_epoch = self._unlabelled.epochs_completed

                validation_loss, validation_reconstruction_loss = \
                    self._feature_network.loss(self._validation.next_batch(self._validation.num_examples)[0])
                reconstruction = self._feature_network.reconstruct(self._validation.next_batch(100)[0])

                for i in range(100):
                    ax = fig.add_subplot(10, 10, i + 1)
                    ax.imshow(reconstruction[i].reshape(28, 28), cmap=cm.gray)

                fig.savefig('reconstruction.png')
                #added logging
                print("{0} Unsupervised Epochs Completed. Training_loss = {3}, Validation loss = {1},"
                      " reconstruction loss = {2}".format(last_epoch, validation_loss, validation_reconstruction_loss, training_loss))

                save_dict[last_epoch%EPOCH_WINDOW_FOR_STOPPING] = (self._save_filename+'_ae_'+str(last_epoch)+'.net', self._feature_network.get_save_state())
                reconstruction_loss_condition = stop_for_reconstruction_loss(validation_reconstruction_loss)
                """
                if self._early_stopping:
                    if stop_for_reconstruction_loss(validation_reconstruction_loss):
                        print('Convergence by Early Stopping Criterion')
                        break
                """
                if reconstruction_loss_condition:
                    print('Convergence by Stopping Criterion. weight_cond = {0}, bias_cond = {1}, recons_cond={2}'.format(
                        1, 1, reconstruction_loss_condition
                    ))
                    break

        for model_number in save_dict:
            filename = save_dict[model_number][0]
            save_state = save_dict[model_number][1]
            self._feature_network.save(filename, save_state)

        self._after_unsupervised_training()

    def build_validation_features(self):
        validation_batch_input, validation_batch_labels = self._validation.next_batch(self._validation.num_examples)
        self._validation_features = self._feature_network.encoding(validation_batch_input)
        self._validation_labels = validation_batch_labels

    def log_loss(self, filename):
        loss_logfile = os.path.join(self._run_folder, filename)
        with open(loss_logfile, 'w') as loss_logfile:
            csvwriter = csv.writer(loss_logfile)
            for row in self.loss_log:
                csvwriter.writerow(row)

    def _after_unsupervised_training(self):
        self.build_validation_features()
        self.log_loss('unsupervised_log.csv')

    @staticmethod
    @gen_utils.ready_generator
    def loss_stopping_criterion():
        stop_cond = False
        prev_loss = yield
        new_loss = yield stop_cond
        while True:
            one_prev_loss = new_loss
            stop_cond = new_loss > prev_loss
            # TODO May need a if breaking (so that generator terminates smoothly?)
            new_loss = yield stop_cond
            prev_loss = one_prev_loss

    def run_supervised_training(self):
        self.loss_log = [('training_loss', 'validation_loss')]
        stop_for = SelfTaughtTrainer.loss_stopping_criterion()
        last_epoch = 0
        validation_loss = 0
        save_dict = {}
        while self._labelled.epochs_completed < self._max_epochs:
            input_batch, output_labels = self._labelled.next_batch(self._batch_size)
            features = self._feature_network.encoding(input_batch)
            training_loss = self._output_network.partial_fit(features, output_labels)
            self.loss_log.append((training_loss, validation_loss))

            if self._labelled.epochs_completed > last_epoch:
                # TODO Reminder: epoch means one run through all data. Data retrieved in order, must be shuffled.
                #       Answer: shuffled by dataset object after every epoch
                last_epoch = self._labelled.epochs_completed
                self._output_network.save(self._save_filename+'_ffd_'+str(last_epoch)+'.net')

                save_dict[last_epoch%100] = (self._save_filename+'_ffd_'+str(last_epoch)+'.net', self._output_network.get_save_state())
                validation_loss = self._output_network.loss_on(self._validation_features, self._validation_labels)
                print('{0} Supervised Epochs Completed. validation loss ={1}'.format(last_epoch, validation_loss))

                loss_stop = stop_for(validation_loss)
                if self._output_network.learning_rate < self._learning_rate_limit and loss_stop:
                        print("Convergence by Stopping Criterion. learning_rate= {0}, loss_stop = {1}".format(
                                                                                            self._output_network.learning_rate,
                                                                                            loss_stop))
                        break

        for model_number in save_dict:
            filename = save_dict[model_number][0]
            save_state = save_dict[model_number][1]
            self._feature_network.save(filename, save_state)

        self._after_supervised_training()

    def run_test_data(self):
        test_batch_input, test_batch_labels = self._test.next_batch(self._test.num_examples)
        self._test_features = self._feature_network.encoding(test_batch_input)
        test_output = self._output_network.encoding(self._test_features)
        self._test_accuracy = SelfTaughtTrainer.accuracy(test_output, test_batch_labels)

    def _after_supervised_training(self):
        self.run_test_data()
        self.log_loss('supervised_log.csv')


    @staticmethod
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])

    @property
    def test_accuracy(self):
        return self._test_accuracy

    def validation_accuracy(self):
        valid_batch_labels = self._validation.next_batch(self._validation.num_examples)[1]
        validation_output = []
        for validation_features in self._validation_features:
            validation_output.append(self._output_network(validation_features))
        validation_output = np.array(validation_output)
        return SelfTaughtTrainer.accuracy(validation_output, valid_batch_labels)

    def output(self, input_batch):
        features = self._feature_network.encoding(input_batch)
        return self._output_network.encoding(features)

if __name__ == '__main__':
    trainer = SelfTaughtTrainer.from_only_labelled(ae.Autoencoder([784, 196], sparse=True, learning_rate=0.001),
                                                   ffd.FeedForwardNetwork([196, 10]),
                                                   100,
                                                   read_data_sets('', one_hot=True),
                                                   save_filename='mnist_self_taught'
                                                   )
    trainer.run_unsupervised_training()
    trainer.run_supervised_training()
    print(trainer._test_accuracy)


