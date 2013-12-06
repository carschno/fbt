__author__ = 'carsten'

import unittest
import numpy as np
from numpy import testing
from collections import Counter

import TrainModel


class MyTestCase(unittest.TestCase):
    def test_generatetermvector(self):
        terms = ["a", "b", "c", "d"]
        doc1 = Counter({"a": 2})
        vector1 = np.array([2, 0, 0, 0])
        doc2 = Counter({"a": 2, "e": 3})

        vector = TrainModel.generate_docvector(doc1, term_list=terms)
        testing.assert_array_equal(vector1, vector)

        vector = TrainModel.generate_docvector(doc2, term_list=terms)
        testing.assert_array_equal(vector1, vector)


if __name__ == '__main__':
    unittest.main()
