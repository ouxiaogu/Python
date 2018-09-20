# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-12 18:28:33

unit test/visualization for Ridge Detection

Last Modified by:  ouxiaogu
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt

import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from RidgeDetector import *
from ImGUI import imshowMultiple
from ImDescriptors import printImageInfo

IMFILE = r'C:\Localdata\D\Note\Python\misc\SEM\samples\Calaveras_v3_1001_averaged.pgm'
# IMFILE = r'C:\Localdata\D\Note\Python\misc\SEM\samples\Calaveras_v3_p1521_regular.bmp'
# IMFILE = r'C:\Users\peyang\github\Canny-edge-detector-master\emilia.jpg'
# IMFILE = r'C:\Users\ouxiaogu\Documents\github\Canny-edge-detector\emilia.jpg'

class TestRD(unittest.TestCase):
    def setUp(self):
        self.imfile = IMFILE

def display(dump_contour=False):
    dt = RidgeDetector(IMFILE, sigma=2, thresL=0.1, thresH=0.4, gapLimit=2, minSegLength=10)
    dt.run()
    dt.cropImagesToBBox()
    diff = dt.gNH ^ dt.imcontour
    imshowMultiple( [dt.im, dt.Ig, dt.Rg_Mag, dt.gN, dt.gNL, dt.gNH, dt.imcontour, diff],
                    ['original', 'Gaussian', 'Ridge Mag', 'Ridge nms', 'Ridge NL', 'Ridge NH', 'contour', 'contour^NH'] )
    imshowMultiple( [dt.Ig, dt.Ig_dx, dt.Ig_dy, dt.Ig_dxdx, dt.Ig_dxdy, dt.Ig_dydy, dt.Rg_Mag, dt.Rg_OrgMag],
                    ['Gaussian', 'Ix', 'Iy', 'Ixx', 'Ixy', 'Iyy', 'Rg_Mag', 'Rg_OrgMag'] )
    
    imshowMultiple( [dt.gNL, dt.gNH, dt.imcontour, diff],
                    ['Ridge NL', 'Ridge NH', 'contour', 'contour^NH'] )
        
    # fig, ax = plt.subplots()
    # ax.imshow(dt.imcontour)
    # plt.title('contour')
    # plt.show()
    
    if dump_contour:
        dt.saveContour('./contour.txt')

def displayMxpResult():
    cwd = r'/gpfs/SQA/FEM/SHARED/regression/MXP/nightly/target/MXP1_job9/h/cache/dummydb/result/MXP/job1/ContourExtraction400result1'
    sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../common")
    from FileUtil import gpfs2WinPath
    from ImGUI import read_pgm
    cwd = gpfs2WinPath(cwd)
    pattern = '1001'
    filenames = ['IG', 'Ig_dx', 'Ig_dy', 'Ig_dxdx', 'Ig_dxdy', 'Ig_dydy', 'RD_Mag', 'RD_OrgMag']
    filenames = [pattern+'_'+n+'.pgm' for n in filenames]
    images = [read_pgm(os.path.join(cwd, imfile)) for imfile in filenames]
    imshowMultiple(images, filenames)

if __name__ == '__main__':
    display(dump_contour=True)
    
    # displayMxpResult()
    
    # unittest.main()
