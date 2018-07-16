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
        im = np.ones(self.shape, dtype=np.uint8)
        ref = equalizeHisto(im)
        print(Histogram(ref))

if __name__ == '__main__':
    unittest.main()