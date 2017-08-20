import numpy as np

from utils import ae
from utils import ffd
import utils.generator as gen_utils
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base

LEARNING_RATE_LIMIT = 0.00001


class SelfTaughtTrainer(object):
    """This class creates a trainer object that will assist in self-taught training of a network.
    The training happens in two step. run_unsupervised_training() trains the network on unlabelled data.
    run_supervised_training() trains the network on labelled data. You need to provide this class with an
    network object."""

    def __init__(self, feature_network, output_network, batch_size,
                 unlabelled, labelled, validation, test, save_filename, max_epochs=20000):
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
        self._learning_rate_limit = LEARNING_RATE_LIMIT
        self._save_filename = save_filename

    @classmethod
    def from_only_labelled(cls, feature_network, output_network, batch_size,
                           data, save_filename, unlabelled_pct=80, max_epochs=20000):
        num_unlabelled = int((unlabelled_pct/100.0)*data.train.num_examples)
        unlabelled_data = DataSet(*data.train.next_batch(num_unlabelled), reshape=False)
        labelled_data = DataSet(*data.train.next_batch(data.train.num_examples-num_unlabelled), reshape=False)
        validation = data.validation
        test = data.test
        return cls(feature_network, output_network, batch_size, unlabelled_data,
                   labelled_data, validation, test, save_filename, max_epochs)

    @staticmethod
    @gen_utils.ready_generator
    def list_of_values_stopping_criterion():    #TODO Not general: only accepts list of matrices
        stop_cond = False
        prev_values = yield
        new_values = yield stop_cond
        while True:
            new_values_temp = new_values
            diff_weights = np.subtract(new_values, prev_values)
            pct_change_values = np.divide(diff_weights, np.array(prev_values))
            max_diff_pct = np.max(np.concatenate([change.flatten() for change in pct_change_values], axis=0))
            stop_cond = max_diff_pct < 1
            # TODO May need a if breaking (so that generator terminates smoothly?)
            new_values = yield stop_cond
            prev_values = new_values_temp

    def run_unsupervised_training(self):
        stop_for_w = SelfTaughtTrainer.list_of_values_stopping_criterion()
        stop_for_b = SelfTaughtTrainer.list_of_values_stopping_criterion()
        last_epoch = 0
        while self._unlabelled.epochs_completed < (self._max_epochs - 19967):
            self._feature_network.partial_fit(self._unlabelled.next_batch(self._batch_size)[0])

            if self._unlabelled.epochs_completed > last_epoch:
                print("{0} Unsupervised Epochs Completed".format(last_epoch))
                last_epoch = self._unlabelled.epochs_completed
                self._feature_network.save(self._save_filename+'_ae_'+str(last_epoch)+'.net')
                weight_condition = stop_for_w(self._feature_network.weights)
                bias_condition = stop_for_b(self._feature_network.biases)
                if weight_condition and bias_condition:
                        print("Convergence by Stopping Criterion")
                        break
        self._after_unsupervised_training()

    def _after_unsupervised_training(self):
        self._validation_features = []
        self._validation_labels = []

        validation_batch_input, validation_batch_labels = self._validation.next_batch(self._validation.num_examples)
        self._validation_features = self._feature_network.encoding(validation_batch_input)
        self._validation_labels = validation_batch_labels


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
        stop_for = SelfTaughtTrainer.loss_stopping_criterion()
        prev_epochs = 0
        while self._labelled.epochs_completed < self._max_epochs:
            input_batch, output_labels = self._labelled.next_batch(self._batch_size)
            features = self._feature_network.encoding(input_batch)
            cost = self._output_network.partial_fit(features, output_labels)

            if self._labelled.epochs_completed > prev_epochs:
                # TODO Reminder: epoch means one run through all data. Data retrieved in order, must be shuffled.
                print(prev_epochs, " Supervised Epochs Completed")
                prev_epochs = self._labelled.epochs_completed
                self._output_network.save(self._save_filename+'_ffd_'+str(prev_epochs)+'.net')
                loss_stop = stop_for(self._output_network.loss_on(self._validation_features, self._validation_labels))
                if self._output_network.learning_rate < self._learning_rate_limit and loss_stop:
                        print("Convergence by Stopping Criterion")
                        break
        self._after_supervised_training()

    def _after_supervised_training(self):
        test_batch_input, test_batch_labels = self._test.next_batch(self._test.num_examples)
        self._test_features = self._feature_network.encoding(test_batch_input)
        test_output = self._output_network.encoding(self._test_features)
        self._test_accuracy = SelfTaughtTrainer.accuracy(test_output, test_batch_labels)

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
    trainer = SelfTaughtTrainer.from_only_labelled(ae.Autoencoder([784, 500, 200]),
                                                   ffd.FeedForwardNetwork([200, 100, 10]),
                                                   100,
                                                   read_data_sets('', one_hot=True),
                                                   save_filename='mnist_self_taught'
                                                   )
    trainer.run_unsupervised_training()
    trainer.run_supervised_training()
    print(trainer._test_accuracy)


