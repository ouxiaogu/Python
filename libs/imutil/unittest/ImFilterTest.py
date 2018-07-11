import math
import unittest
import numpy as np

import sys
import os.path
sys.path.append(os.path.abspath("../"))
from ImFilters import *

class TestFilters(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()