import src.feature_network as source
import numpy as np
import unittest

class FeatureNetworkTests(unittest.TestCase):

    def test_get_features(self):
        batch_size = 100
        input_size = 500
        input_batch = np.zeros([batch_size, input_size])
        features = source.get_features(input_batch, model)
