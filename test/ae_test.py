import unittest
import src.utils.ae as source
import numpy as np

class AutoencoderTest(unittest.TestCase):

    def test_serialization(self):
        if True:
            autoenc = source.Autoencoder([784, 500, 200])
            inp1 = np.ndarray(shape=(100,784))
            inp2 = np.ndarray(shape=(100,784))
            autoenc.partial_fit(inp1)
            autoenc.partial_fit(inp2)
            weights = autoenc.weights
            autoenc.stop_session()

        if True:
            autoenc1 = source.Autoencoder([784, 500, 100])
            autoenc1.partial_fit(inp1)
            filename = 'ae_500_200.pnt'
            source.Autoencoder.save_pickle(autoenc1, filename)
            model = source.Autoencoder.load_pickle(filename)
            model.partial_fit(inp2)

        weights_after = model.weights

        self.assertListEqual(weights, weights_after)
