import math
import unittest
import numpy as np
from collections import OrderedDict

import sys
import os.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../")
from ImDescriptors import *

class TestImDescriptors(unittest.TestCase):
    def setUp(self):
        self.shape = (2, 3)
        self.im = np.arange(6, dtype=np.uint8).reshape(self.shape )

    def test_calcHist(self):
        bas = np.ones(6, dtype=np.uint8)
        ref = calcHist(self.im)
        np.testing.assert_equal(bas, ref[0:6])

    def test_calcHist_dict(self):
        self.im[1, :] = [0, 1, 2]
        bas = OrderedDict(zip([0, 1, 2], [2, 2, 2]) )
        ref = calcHist(self.im, hist_type='OrderedDict')
        np.testing.assert_equal(bas, ref)

    def test_cdfHisto(self):
        print("cdfHisto")
        hist = np.arange(10)
        print(cdfHisto(hist) )

    def test_cdfHisto_dict(self):
        arr = np.array( [[0, 0, 0], [0, 227, 227], [0, 227, 227] ], dtype=np.uint8)
        hist = calcHist(arr, hist_type='OrderedDict')
        np.testing.assert_equal(hist, OrderedDict([(0, 5), (227, 4)]))
        print(cdfHisto(hist))

    def test_addOrdereddDict(self):
        a = OrderedDict([(0, 5), (227, 4), (371, 55)])
        b = OrderedDict([(0, 141), (2, 3), (227, 255)])
        bas = OrderedDict([(0, 146), (2, 3), (227, 259), (371, 55)])
        ref = addOrdereddDict(a, b)
        np.testing.assert_equal(bas, ref)

    def test_subOrdereddDict(self):
        a = OrderedDict([(0, 5), (227, 4), (371, 55)])
        b = OrderedDict([(0, 2), (227, 3)])
        ref = subOrdereddDict(a, b)
        np.testing.assert_equal(OrderedDict([(0, 3), (227, 1), (371, 55)]), ref)

if __name__ == '__main__':
    unittest.main()