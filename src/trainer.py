import numpy as np
import datetime as dt
import os
import csv

from src.utils import ae
from src.utils import ffd
import src.utils.generator as gen_utils
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

tf.logging.set_verbosity(tf.logging.INFO)
LEARNING_RATE_LIMIT = 0.00001


class SelfTaughtTrainer(object):
    """This class creates a trainer object that will assist in self-taught training of a network.
    The training happens in two step. run_unsupervised_training() trains the network on unlabelled data.
    run_supervised_training() trains the network on labelled data. You need to provide this class with an
    network object."""

    def __init__(self, feature_network, output_network, batch_size,
                 unlabelled, labelled, validation, test, save_filename, validation_lab=None, run_folder=None,
                 early_stopping=True, epoch_window_for_stopping=100, max_epochs=20000):
        self._unlabelled = unlabelled
        print("Unlabelled Examples", self._unlabelled.num_examples if self._unlabelled else 0)
        self._labelled = labelled
        print("Labelled Examples", self._labelled.num_examples if self._labelled else 0)
        self._validation = validation
        print("Validation Examples", self._validation.num_examples if self._validation else 0)
        self._validation_lab = validation_lab
        if validation_lab is not None:
            print("Labelled Validation Examples", self._validation_lab.num_examples if self._validation_lab else 0)
        self._test = test
        print("Test Examples", self._test.num_examples)
        self._feature_network = feature_network
        self._output_network = output_network
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._early_stopping = early_stopping
        self._epoch_window_for_stopping = epoch_window_for_stopping
        self._learning_rate_limit = LEARNING_RATE_LIMIT
        self._run_folder = run_folder
        if run_folder is None:
            timestamp = str(dt.datetime.now())
            timestamp = timestamp.replace(' ', '_').replace(':', '-').replace('.', '-')
            project_path = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
            self._run_folder = os.path.join(project_path, 'results', timestamp)

        if not os.path.exists(self._run_folder):
            os.mkdir(self._run_folder)
        self._save_filename = os.path.join(self._run_folder, save_filename)

        self.save_dict={}

    @classmethod
    def from_only_labelled(cls, feature_network, output_network, batch_size,
                           data, save_filename, run_folder=str(dt.datetime.now()), early_stopping=True,
                           epoch_window_for_stopping=25, unlabelled_pct=80, max_epochs=20000):
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
                   labelled_data, validation, test, save_filename, run_folder,
                   early_stopping, epoch_window_for_stopping, max_epochs)

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
    def historic_change_stopping_criterion(window):
        epoch_window = window
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
    def early_stopping_criterion(window):
        patience_window = window
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
        stop_for_reconstruction_loss = SelfTaughtTrainer.early_stopping_criterion(self._epoch_window_for_stopping)
        last_epoch = 0
        validation_loss = 0
        validation_reconstruction_loss = 0
        start_flag = True
        try:
            while self._unlabelled.epochs_completed < self._max_epochs:
                training_loss = self._feature_network.partial_fit(self._unlabelled.next_batch(self._batch_size)[0])
                self.loss_log.append((training_loss, validation_loss, validation_reconstruction_loss))

                if self._unlabelled.epochs_completed > last_epoch or start_flag:
                    start_flag = False
                    last_epoch = self._unlabelled.epochs_completed

                    validation_loss, validation_reconstruction_loss = \
                        self._feature_network.loss(self._validation.next_batch(self._validation.num_examples)[0])
                    reconstruction = self._feature_network.reconstruct(self._validation.next_batch(100)[0])

                    print("{0} Unsupervised Epochs Completed. Training_loss = {3}, Validation loss = {1},"
                          " reconstruction loss = {2}".format(last_epoch, validation_loss, validation_reconstruction_loss, training_loss))

                    self.save_dict[last_epoch % self._epoch_window_for_stopping] = (self._save_filename+'_ae_'+str(last_epoch)+'.net', self._feature_network.get_save_state())
                    reconstruction_loss_condition = stop_for_reconstruction_loss(validation_reconstruction_loss)
                    """
                    if self._early_stopping:
                        if stop_for_reconstruction_loss(validation_reconstruction_loss):
                            print('Convergence by Early Stopping Criterion')
                            break
                    """
                    if reconstruction_loss_condition:
                        print('Convergence by Stopping Criterion. recons_cond={0}'.format(reconstruction_loss_condition))
                        break
        finally:
            try:
                self._after_unsupervised_training()
            finally:
                self.save_feature_models()
        return validation_reconstruction_loss

    def save_feature_models(self):
        for model_number in self.save_dict:
            filename = self.save_dict[model_number][0]
            save_state = self.save_dict[model_number][1]
            self._feature_network.save(filename, save_state)

    def build_validation_features(self):
        validation_batch_input, validation_batch_labels = self._validation_lab.next_batch(self._validation_lab.num_examples)
        if self._feature_network is None:
            self._validation_features = validation_batch_input
        else:
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
        #self.log_loss('unsupervised_log.csv')
        self.save_feature_models()

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
        stop_for = SelfTaughtTrainer.early_stopping_criterion(5)
        last_epoch = 0
        validation_loss = 0
        save_dict = {}
        while self._labelled.epochs_completed < self._max_epochs:
            input_batch, output_labels = self._labelled.next_batch(self._batch_size)
            if self._feature_network is None:
                features = input_batch
            else:
                features = self._feature_network.encoding(input_batch)
            training_loss = self._output_network.partial_fit(features, output_labels)
            self.loss_log.append((training_loss, validation_loss))

            if self._labelled.epochs_completed > last_epoch:
                # TODO Reminder: epoch means one run through all data. Data retrieved in order, must be shuffled.
                #       Answer: shuffled by dataset object after every epoch
                last_epoch = self._labelled.epochs_completed
                self._output_network.save(self._save_filename+'_ffd_'+str(last_epoch)+'.net')

                save_dict[last_epoch%self._epoch_window_for_stopping] = \
                    (self._save_filename+'_ffd_'+str(last_epoch)+'.net', self._output_network.get_save_state())

                validation_loss = self._output_network.loss_on(self._validation_features, self._validation_labels)
                valid_acc = self.get_accuracy_on(self._validation_lab)
                print('{0} Supervised Epochs Completed. validation acc ={1}'.format(last_epoch, valid_acc))

                validation_acc_stop = stop_for(1/valid_acc)
                if validation_acc_stop:
                        print("Convergence by Stopping Criterion. learning_rate= {0}, loss_stop = {1}".format(
                                                                                            self._output_network.learning_rate,
                                                                                            validation_acc_stop))
                        break

        for model_number in save_dict:
            filename = save_dict[model_number][0]
            save_state = save_dict[model_number][1]
            self._feature_network.save(filename, save_state)

        self._after_supervised_training()
        return self._test_accuracy

    def get_accuracy_on(self, dataset):
        test_batch_input, test_batch_labels = dataset.next_batch(dataset.num_examples)
        if self._feature_network is None:
            self._test_features = test_batch_input
        else:
            self._test_features = self._feature_network.encoding(test_batch_input)
        test_output = self._output_network.encoding(self._test_features)
        return SelfTaughtTrainer.accuracy(test_output, test_batch_labels)

    def _after_supervised_training(self):
        self._test_accuracy = self.get_accuracy_on(self._test)
        self.log_loss('supervised_log.csv')


    @staticmethod
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])

    @property
    def test_accuracy(self):
        return self._test_accuracy

    def output(self, input_batch):
        features = self._feature_network.encoding(input_batch)
        return self._output_network.encoding(features)

if __name__ == '__main__':
    timestamp = str(dt.datetime.now())
    timestamp = timestamp.replace(' ', '_').replace(':', '-').replace('.', '-')
    run_folder = os.path.join(os.path.pardir, 'results', timestamp)
    trainer = SelfTaughtTrainer.from_only_labelled(ae.Autoencoder([784, 196], beta=0.1, sparse=True, sparsity=0.10,
                                                                  lambda_=0.03, learning_rate=0.01, logdir=run_folder),
                                                   ffd.FeedForwardNetwork([196, 10], dynamic_learning_rate=False),
                                                   100,
                                                   read_data_sets('', one_hot=True),
                                                   save_filename='mnist_self_taught',
                                                   run_folder=run_folder
                                                   )
    '''
    root = os.path.join(os.path.pardir, 'results', '2017-09-20_19-22-13-311688')
    auto_file = os.path.join(root, 'mnist_self_taught_ae_48.net')
    supe_file = os.path.join(root, 'mnist_self_taught_ffd_261.net')
    auto = ae.Autoencoder.load_model(auto_file, logdir=run_folder)  # type: ae.Autoencoder
    """:type : ae.Autoencoder"""
    supe = ffd.FeedForwardNetwork.load_model(supe_file)
    data = read_data_sets('', one_hot=True)
    trainer = SelfTaughtTrainer.from_only_labelled(auto, ffd.FeedForwardNetwork([196,10]),
                                                   100,
                                                   data,
                                                   save_filename='mnist_self_taught',
                                                   run_folder=run_folder)
    '''
    print(trainer.run_test_data())
    trainer.run_unsupervised_training()
    trainer.run_supervised_training()
    print(trainer._test_accuracy)

