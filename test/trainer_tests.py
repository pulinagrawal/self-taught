import unittest
import numpy as np
import trainer as source

class TrainerTests(unittest.TestCase):
    def test_list_of_values_stopping_criterion(self):
        stop_for_values = source.SelfTaughtTrainer.list_of_values_stopping_criterion()
        values1 = [np.array([[9,8],[7,6]]),np.array([[2,3]])]
        values2 = [np.array([[6,7],[8,11]]),np.array([[2,3]])]
        stop_for_values(values1)
        self.assertFalse(stop_for_values(values2))

if __name__ == '__main__':
    unittest.main()