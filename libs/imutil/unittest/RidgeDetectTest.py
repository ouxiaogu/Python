# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-12 18:28:33



Last Modified by:  ouxiaogu
"""

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
from RidgeDetector import *
from ImGUI import imshowMultiple
from ImDescriptors import printImageInfo

IMFILE = r'C:\Localdata\D\Note\Python\misc\SEM\samples\Calaveras_v3_p1521_regular.bmp'
# IMFILE = r'C:\Users\peyang\github\Canny-edge-detector-master\emilia.jpg'
# IMFILE = r'C:\Users\ouxiaogu\Documents\github\Canny-edge-detector\emilia.jpg'

class TestRD(unittest.TestCase):
    def setUp(self):
        self.imfile = IMFILE

def display(dump_contour=False):
    cim = cv2.imread(IMFILE, 1)
    im = cv2.cvtColor(cim, cv2.COLOR_BGR2GRAY)
    dt = RidgeDetector(im, sigma=2, thresL=0.2, thresH=0.6, gapLimit=2, minSegLength=10)
    dt.run()
    diff = dt.gNH ^ dt.imcontour
    imshowMultiple( [im, dt.Ig, dt.Rg_Mag, dt.gN, dt.gNL, dt.gNH, dt.imcontour, diff],
                    ['original', 'Gaussian', 'Ridge Mag', 'Ridge nms', 'Ridge NL', 'Ridge NH', 'contour', 'diff NH'] )
    fig, ax = plt.subplots()
    ax.imshow(dt.imcontour)
    plt.title('contour')
    plt.show()
    
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
