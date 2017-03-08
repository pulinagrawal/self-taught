import numpy as np

from utils import ae
import generator as myutils
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base


class SelfTaughtTrainer(object):
    """This class creates a trainer object that will assist in self-taught training of a network.
    The training happens in two step. run_unsupervised_training() trains the network on unlabelled data.
    run_supervised_training() trains the network on labelled data. You need to provide this class with an
    network object."""

    def __init__(self, network: ae.Autoencoder, unlabelled, labelled, validation, test, max_epochs=50000):
        self._unlabelled = unlabelled
        self._labelled = labelled
        self._validation = validation
        self._test = test
        self._network = network
        self._max_epochs = max_epochs

    @classmethod
    def from_only_labelled(cls, network, data: base.Dataset, unlabelled_pct=80):
        num_unlabelled = (unlabelled_pct//100*len(data.train.num_examples))
        unlabelled_data = DataSet(*data.train.next_batch(num_unlabelled))
        labelled_data = DataSet(*data.train.next_batch(data.train.num_examples-num_unlabelled))
        validation = data.validation
        test = data.test
        return cls(network, unlabelled_data, labelled_data, validation, test)

    @staticmethod
    @myutils.get_ready
    def value_list_criterion():
        stop_cond = False
        prev_weights = yield
        new_weights = yield stop_cond
        while stop_cond:
            one_prev_weights = new_weights
            stop_cond = np.max(np.divide(np.subtract(new_weights, prev_weights), prev_weights)) > 0.001
            new_weights = yield stop_cond
            prev_weights = one_prev_weights

    def run_unsupervised_training(self):
        stop_for_w = SelfTaughtTrainer.value_list_criterion()
        stop_for_b = SelfTaughtTrainer.value_list_criterion()
        prev_epochs = 0
        while self._unlabelled.epochs_completed < self._max_epochs:
            self._network.partial_fit(self._unlabelled.next_batch(self._network.batch_size))

            if self._unlabelled.epochs_completed > prev_epochs:
                prev_epochs = self._unlabelled.epochs_completed
                self._network.save(self._save_filename+'_'+prev_epochs+'.net')
                if stop_for_w(self._network.weights) and stop_for_b(self._network.biases) is True:
                    break

    def run_supervised_training(self):
        pass

trainer = SelfTaughtTrainer(ae.Autoencoder([784, 500, 200]), read_data_sets(''))
trainer.unsupervised_train()

