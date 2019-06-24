import src.utils.ffd as source
import tensorflow as tf
import src.utils.ae as ae
import src.trainer as trainer
from tensorflow.contrib.learn.python.learn.datasets import mnist
import numpy as np
import unittest

class ClassifierTests(unittest.TestCase):

    def test_mnist_classifier_onehot(self):
        batch_size = 100
        input_size = 784
        feature_network = ae.Autoencoder.get_identity_encoder(input_size)
        network = source.FeedForwardNetwork([input_size, 256, 10])
        datasets = mnist.read_data_sets('', one_hot=True)

        classifier = trainer.SelfTaughtTrainer(feature_network, network, 100, None,
                                               datasets.train, None, datasets.test,
                                               'test_mnist_classifier', validation_lab=datasets.validation)
        classifier._after_unsupervised_training()
        assert classifier.run_supervised_training()>50
