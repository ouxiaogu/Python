# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-10-29 22:18:25

Unit test for SEMContour class

Last Modified by: ouxiaogu
"""

import unittest
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
from SEMContour import *

CONTOURFILE = r'..\..\..\apps\MXP\ContourSelect\samplejob\h\cache\dummydb\result\MXP\job1\ContourExtraction400\461_image_contour.txt'

class TestContour(unittest.TestCase):
    def setUp(self):
        self.contourfile = CONTOURFILE
        contour = SEMContour()
        contour.parseFile(self.contourfile)
        self.contour = contour
        self.contourdf = contour.toDf()

    def test_toDf(self):
        np.testing.assert_array_equal(self.contourdf.columns[1:], self.contour.getColumnTitle())
        grouped = self.contourdf.groupby('polygonId')
        npolygons = self.contour.polygonNum
        self.assertEqual(len(grouped), npolygons)
        
        totalPointsNum = 0
        for i in range(len(self.contour.getPolygonData())):
            totalPointsNum += self.contour.getPolygonData()[i]['vertexNum']
        self.assertEqual(len(self.contourdf), totalPointsNum)    
        
        np.testing.assert_equal(grouped.get_group(0).loc[:, self.contourdf.columns[1:]].values, self.contour.getPolygonData()[0]['points'])
        
        np.testing.assert_equal(grouped.get_group(npolygons-1).loc[:, self.contourdf.columns[1:]].values, self.contour.getPolygonData()[-1]['points'])
        
    def test_clone(self):
        contourclone = self.contour.clone()

        np.testing.assert_equal(self.contourdf.values, contourclone.toDf().values)

    def test_fromDf(self):
        df = self.contourdf.copy()
        df.loc[:, 'test'] = 1
        contour2 = self.contour.fromDf(df)
        contourdf2 = contour2.toDf()
        self.assertEqual(len(self.contourdf), len(contourdf2))
        np.testing.assert_equal(self.contourdf.values, contourdf2.values[:,:-1])

if __name__ == '__main__':
    unittest.main()