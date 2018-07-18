# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-07-18 18:33:04

Unit test for spatial domain filters

Last Modified by: ouxiaogu
"""

import math
import unittest
import numpy as np

import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from SpatialFlt import *

class TestFilters(unittest.TestCase):
    def setUp(self):
        self.shape = (5, 5)

    def test_GaussianFilter(self):
        print(GaussianFilter(self.shape) )

    def test_LaplaceFilter(self):
        print(LaplaceFilter(self.shape) )
        print(LaplaceFilter(self.shape, True) )


if __name__ == '__main__':
    unittest.main()