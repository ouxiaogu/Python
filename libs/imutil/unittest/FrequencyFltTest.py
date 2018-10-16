import math
import unittest
import numpy as np

import sys
import os.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
from FrequencyFlt import *

class TestFreqFilters(unittest.TestCase):
    def setUp(self):
        self.shape = (5, 5)
        self.D0 = 2

    def test_ILPF(self):
        a = ILPF(self.shape, self.D0)
        self.assertAlmostEqual(a[2, 2], 1)
        self.assertAlmostEqual(a[2, 1], 1)
        print(a)

    def test_BLPF(self):
        a = BLPF(self.shape, self.D0, 2)
        self.assertAlmostEqual(a[2, 2], 1)
        self.assertTrue(a[2, 1] < 1)
        print(a)

    def test_GLPF(self):
        a = GLPF(self.shape, self.D0)
        self.assertAlmostEqual(a[2, 2], 1)
        self.assertTrue(a[2, 1] < 1)
        print(a)

    def test_dummy(self):
        funcs = [ILPF, BLPF, GLPF]
        print([f.__name__ for f in funcs])

    def test_mat(self):
        shape = (3, 4)
        nrows, ncols = shape
        ref = np.zeros(shape)
        for y in range(nrows):
            for x in range(ncols):
                ref[y, x] = (-1)**(x+y)

        x, y = np.meshgrid(np.arange(ncols), np.arange(nrows))
        bas = (-1)**(x + y)
        np.testing.assert_equal(bas, ref)

if __name__ == '__main__':
    unittest.main()