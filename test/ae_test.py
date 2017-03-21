import unittest
import ae

class AutoencoderTest(unittest.TestCase):

    def setUp(self):
    def test__create_network1(self):
        network_arch = [784, 100]
        sae = ae.Autoencoder(network_arch)
        sae._layers

    def test__create_network2(self):
        network_arch = [784, 100, 50]
        self.sae = Autoencoder()
