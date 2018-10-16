import math
import unittest
import numpy as np
from collections import OrderedDict

import sys
import os.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../")
from ImTransform import *
from ImDescriptors import *

class TestImTransform(unittest.TestCase):
    def setUp(self):
        self.shape = (2, 3)
        self.im = np.arange(6, dtype=np.uint8).reshape(self.shape )

    def test_equalize(self):
        print("equalizeHisto")
        im = np.ones(self.shape, dtype=np.uint8)
        hist = calcHist(im)
        print(cdfHisto(hist)[:5] )
        ref = equalizeHisto(im, dtype=np.uint8)
        print(calcHist(ref)[:5] )

    def test_dummy(self):
        a = [(1, 2)]
        for i, kv in enumerate(a):
            print(i,kv)

if __name__ == '__main__':
    unittest.main()