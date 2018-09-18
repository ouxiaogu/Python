# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-07-24 10:56:15

GUI function test

Last Modified by:  ouxiaogu
"""

import unittest
import numpy as np

import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../")
from ImGUI import *
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../common")
import logger
log = logger.setup(level='debug')


IMFILE = r'C:\Localdata\D\Note\Python\misc\SEM\samples\Calaveras_v3_1001_averaged.pgm'

class TestImGUI(unittest.TestCase):
    @unittest.skip("temporarily disabled")
    def test_getPolyROI(self):
        row = np.arange(7)
        im = np.tile(row, (7,1))
        log.debug('{}\n'.format(str(im)))
        vertexes = [(1, 2), (1,4), (5,3)] # (x, y)
        # bas = getPolyROI(im, vertexes)
        ref = getPolyROI(im, vertexes)
        log.debug('{}\n'.format(str(ref)))

    def test_getBBox(self):
        bbox = readBBox(IMFILE)
        print('bbox: {}'.format(bbox))

if __name__ == '__main__':
    unittest.main()