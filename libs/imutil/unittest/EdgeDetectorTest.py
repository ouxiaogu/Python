# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-09 20:52:17

unit test/visualization for Edge Detection

Last Modified by: ouxiaogu
"""

import math
import unittest
import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from EdgeDetector import *
from ImGUI import imshowMultiple
from ImDescriptors import printImageInfo

# IMFILE = r'D:\code\Python\apps\MXP\samples\Calaveras_v3_p1521_regular.bmp'
IMFILE = r'C:\Users\ouxiaogu\Documents\github\Canny-edge-detector\emilia.jpg'

class TestFreqFilters(unittest.TestCase):
    def setUp(self):
        self.imfile = IMFILE

def display():
    im = cv2.imread(IMFILE, 0)
    detector = EdgeDetector(im, 0.8, 5, 0.1, 0.3)
    print()
    detector.run()
    imshowMultiple( [detector.im, detector.gim, detector.G, 
                    detector.gN, detector.gcontour],
                    ['original', 'Gaussian', 'Gradient', 'Gradient nms', 'contour']
        )

if __name__ == '__main__':
    display()
    # unittest.main()
    