# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-06 23:50:24



Last Modified by: ouxiaogu
"""
import numpy as np
import unittest

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from ImFeatures import *

class TestImFeatures(unittest.TestCase):
    def test_gradient(self):
        np.random.seed(0)
        im = np.random.randn(3, 3)
        G, theta = gradient(im)
        print("gradient magnitude & theta\n{}\n{}".format(G, theta))

if __name__ == '__main__':
    unittest.main()