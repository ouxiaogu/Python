import math
import unittest
import numpy as np

import sys
import os.path
sys.path.append(os.path.abspath("../"))
from filters import *

import sympy
from sympy import Matrix, symbols, pprint

class TestFilters(unittest.TestCase):

    def test_gaussian_filter(self):
        sigma = 2
        x = symbols('x') # n=1
        G = 1./(sympy.sqrt(2. * sympy.pi)* sigma) * sympy.exp(-x**2/(2.*sigma**2))
        dG = G.diff(x)
        ddG = dG.diff(x)

        flt_G = gaussian_filter(sigma, 0)
        flt_dG = gaussian_filter(sigma, 1)
        flt_ddG = gaussian_filter(sigma, 2)
        hlFltSz = (len(flt_G) - 1)/2

        for ix, val in enumerate(flt_G):
            G_x = G.subs(x, ix - hlFltSz).evalf()
            self.assertAlmostEqual(G_x, val)

        for ix, val in enumerate(flt_G):
            dG_x = dG.subs(x, ix - hlFltSz).evalf()
            self.assertAlmostEqual(dG_x, flt_dG[ix], msg="%f != %f within 7 places at index %d" % (dG_x, flt_dG[ix], ix - hlFltSz))

        for ix, val in enumerate(flt_G):
            ddG_x = ddG.subs(x, ix - hlFltSz).evalf()
            self.assertAlmostEqual(ddG_x, flt_ddG[ix])

    def test_padding(self):
        padsize = 1
        padval = 0.
        inarr = np.empty((3,) )
        ouarr = padding(inarr, padsize, padval)
        self.assertAlmostEqual(inarr.flatten().tolist(), ouarr[padsize:(padsize+3)].flatten().tolist() )
        self.assertAlmostEqual(padval, ouarr[0])

        padsize = 2
        padval = 3.
        inarr = np.empty((3, 1) )
        ouarr = padding(inarr, padsize, padval)
        self.assertEqual(inarr.flatten().tolist(), ouarr[padsize:(padsize+3), 0].flatten().tolist() )
        # self.assertAlmostEqual(padval, ouarr[0])

        padsize = 2
        padval = 5.
        inarr = np.empty((2, 3) )
        ouarr = padding(inarr, padsize, padval)
        self.assertEqual(inarr.tolist(), ouarr[padsize:(padsize+2), padsize:(padsize+3)].tolist())
        self.assertAlmostEqual(padval, ouarr[-1, -1])

if __name__ == '__main__':
    unittest.main()