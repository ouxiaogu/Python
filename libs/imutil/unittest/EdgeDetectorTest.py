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

# IMFILE = r'C:\Users\peyang\github\Canny-edge-detector-master\emilia.jpg'
IMFILE = r'C:\Users\ouxiaogu\Documents\github\Canny-edge-detector\emilia.jpg'

class TestFreqFilters(unittest.TestCase):
    def setUp(self):
        self.imfile = IMFILE

def display(dump_contour=False):
    cim = cv2.imread(IMFILE, 1)
    im = cv2.cvtColor(cim, cv2.COLOR_BGR2GRAY)
    dt = EdgeDetector(im, 0.6, 5, 0.1, 0.35)
    dt.run()
    diff = dt.gNH ^ dt.gcontour
    imshowMultiple( [im, dt.gim, dt.G, dt.gN, dt.gNL, dt.gNH, dt.gcontour, diff],
                    ['original', 'Gaussian', 'Gradient', 'Gradient nms', 'Gradient NL', 'Gradient NH', 'contour', 'diff NH'] )
    imshowMultiple([cim, dt.gcontour], ['original', 'contour'])
    if dump_contour:
        with open("./contour.txt", 'w+') as fout:
            header = dt.attrs
            header.insert(0, 'segId')
            print(type(header), header)
            fout.write('\t'.join(header ) + '\n')
            formater = ["{}" for i in range(len(header))]
            formater = '\t'.join(formater)
            formater += '\n'
            for i, seg in enumerate(dt.contour):
                for point in seg:
                    point = list(point)
                    point.insert(0, i)
                    fout.write(formater.format(*point))

if __name__ == '__main__':
    display()
    # unittest.main()
