# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-09 20:52:17

unit test/visualization for Edge Detection

Last Modified by:  ouxiaogu
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
import os.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
from EdgeDetector import *
from ImGUI import imshowMultiple
from ImDescriptors import printImageInfo

# IMFILE = r'C:\Localdata\D\Note\Python\misc\SEM\samples\Calaveras_v3_1001_averaged.pgm'
# IMFILE = r'C:\Localdata\D\Note\Python\misc\SEM\samples\Calaveras_v3_p1521_regular.bmp'
# IMFILE = r'C:\Users\peyang\github\Canny-edge-detector-master\emilia.jpg'
IMFILE = r'C:\Users\ouxiaogu\Documents\github\Canny-edge-detector\emilia.jpg'

class TestED(unittest.TestCase):
    def setUp(self):
        self.imfile = IMFILE

    def test_decideAngleType(self):
        self.assertEqual(decideAngleType(-158), 0)
        self.assertEqual(decideAngleType(158), 0)

        self.assertEqual(decideAngleType(-157), 1)
        self.assertEqual(decideAngleType(23), 1)


def display(dump_contour=False):
    im = cv2.imread(IMFILE, 0)
    # im = cv2.cvtColor(cim, cv2.COLOR_BGR2GRAY)
    dt = EdgeDetector(im, sigma=2, ksize=None, thresL=0.05, thresH=0.15, gapLimit=2, minSegLength=10)
    dt.run()
    diff = dt.gNH ^ dt.imcontour
    imshowMultiple( [im, dt.gim, dt.G, dt.gN, dt.gNL, dt.gNH, dt.imcontour, diff],
                    ['original', 'Gaussian', 'Gradient', 'Gradient nms', 'Gradient NL', 'Gradient NH', 'contour', 'contour^NH'] )
    fig, ax = plt.subplots()
    ax.imshow(dt.imcontour)
    plt.title('contour')
    plt.show()
    
    if dump_contour:
        with open("./contour.txt", 'w+') as fout:
            header = dt.attrs
            header.insert(0, 'polygonId')
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
    
    #unittest.main()
