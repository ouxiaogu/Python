# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-06 23:50:24



Last Modified by: ouxiaogu
"""
import numpy as np
import unittest
import cv2

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from ImFeatures import *
from ImGUI import imshowMultiple
from ImDescriptors import printImageInfo

IMFILE = r'C:\Users\ouxiaogu\Documents\github\Canny-edge-detector\emilia.jpg'

class TestImFeatures(unittest.TestCase):
    def test_gradient(self):
        np.random.seed(0)
        im = np.random.randn(3, 3)
        G, theta = gradient(im)
        print("gradient magnitude & theta\n{}\n{}".format(G, theta))

def display():
    im = cv2.imread(IMFILE)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    Gx, Gy = gradientXY(gray)
    G, _ = gradient(gray)
    printImageInfo(Gx)
    printImageInfo(Gy)
    printImageInfo(G)
    imshowMultiple([im, Gx, Gy, G],
        ['original', 'Gx', 'Gy', 'G'])


if __name__ == '__main__':
    unittest.main()
    display()