import math
import unittest
import numpy as np

import sys
import os.path
sys.path.append(os.path.abspath("../"))
from ImTransform import *

class TestImTransform(unittest.TestCase):
    def setUp(self):
        self.shape = (2, 3)
        self.im = np.arange(6, dtype=np.uint8).reshape(self.shape )

    def test_histogram(self):
        bas = np.ones(6, dtype=np.uint8)
        ref = Histogram(self.im)
        np.testing.assert_equal(bas, ref[0:6])

    def test_equalize(self):
        print("equalizeHisto")
        im = np.ones(self.shape, dtype=np.uint8)
        hist = Histogram(im)
        print(cdfHisto(hist)[:5] )
        ref = equalizeHisto(im)
        print(Histogram(ref)[:5] )

    def test_cdfHisto(self):
        print("cdfHisto")
        hist = np.arange(10)
        print(cdfHisto(hist) )


if __name__ == '__main__':
    unittest.main()