import math
import unittest
import numpy as np

import sys
import os.path
sys.path.append(os.path.abspath("../"))
from filters import *

import sympy
from sympy import Matrix, symbols, pprint

def random_arrary(length, shape=None):
    arr = np.arange(length)
    np.random.shuffle(arr)
    if shape is not None and isinstance(shape, tuple):
        arr = arr.reshape(shape)
    return arr

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
            self.assertAlmostEqual(G_x, val, 2)

        for ix, val in enumerate(flt_G):
            dG_x = dG.subs(x, ix - hlFltSz).evalf()
            self.assertAlmostEqual(dG_x, flt_dG[ix], 2, msg="%f != %f within 7 places at index %d" % (dG_x, flt_dG[ix], ix - hlFltSz))

        for ix, val in enumerate(flt_G):
            ddG_x = ddG.subs(x, ix - hlFltSz).evalf()
            self.assertAlmostEqual(ddG_x, flt_ddG[ix], 2)

        np.testing.assert_equal(gaussian_filter(0.5, dtype=np.int32), [1, 4, 6, 4, 1])

    def test_cv_gaussian_kernel(self):
        print(cv_gaussian_kernel(3, dtype=np.int32))

    def test_padding(self):
        padsize = 1
        padval = 0.
        src = random_arrary(3, (3,))
        dst = padding(src, padsize, padval)
        self.assertEqual((5,), dst.shape)
        self.assertAlmostEqual(src.flatten().tolist(), dst[padsize:(padsize+3)].flatten().tolist() )
        self.assertAlmostEqual(padval, dst[0])

        padsize = 2
        padval = 3.
        src = random_arrary(3, (3, 1))
        dst = padding(src, padsize, padval)
        self.assertEqual((7, 1), dst.shape)
        self.assertEqual(src.flatten().tolist(), dst[padsize:(padsize+3), 0].flatten().tolist() )
        # self.assertAlmostEqual(padval, dst[0])

        padsize = 3
        padval = 3.
        src = random_arrary(3, (1, 3))
        dst = padding(src, padsize, padval)
        self.assertEqual((1, 9), dst.shape)
        self.assertEqual(src.flatten().tolist(), dst[0, padsize:(padsize+3)].flatten().tolist() )
        # self.assertAlmostEqual(padval, dst[0])

        padsize = 2
        padval = 5.
        src = random_arrary(6, (2, 3))
        dst = padding(src, padsize, padval)
        self.assertEqual((6, 7), dst.shape)
        self.assertEqual(src.tolist(), dst[padsize:(padsize+2), padsize:(padsize+3)].tolist())
        self.assertAlmostEqual(padval, dst[-1, -1])

    def test_padding_backward(self):
        a = np.random.rand(4, 3)
        padshape = (1, 3)
        res = padding_backward(a, padshape)
        myslice = tuple(slice(0, sz) for sz in a.shape)
        np.testing.assert_almost_equal(a, res[myslice])
        bas = np.zeros(padshape)
        np.testing.assert_almost_equal(bas, res[-1:, -3:])

    def test_convolve1D(self):
        src = random_arrary(3, (1, 3))
        flt_G = gaussian_filter(1, 0) # len=9
        hlFltSz = (len(flt_G) - 1)//2  # 4
        ref = convolve(src, flt_G)

        bas = np.convolve(src[0, :], flt_G, "same")
        bas = bas[(hlFltSz-1):(hlFltSz+2)] # extend size of src from bas center
        self.assertEqual(bas.tolist(), ref[0, :].tolist() )

    def test_convolve2D(self):
        src = random_arrary(81, (9, 9))
        # src = np.random.randn(9, 9)

        flt_G = np.asarray(gaussian_filter(1, 0))
        hlFltSz = (len(flt_G) - 1)/2
        print(flt_G)
        flt_Gy = cv_gaussian_kernel(3)
        print(flt_Gy)
        ref = convolve(src, flt_G, flt_Gy)

        tmp = np.zeros(src.shape)
        for i in range(9):
            tmp[i, :] = np.convolve(src[i, :], flt_G, "same")
        bas = np.zeros(src.shape)
        for j in range(9):
            bas[:, j] = np.convolve(tmp[:, j], flt_Gy, "same")
        np.testing.assert_almost_equal(bas, ref, decimal=5)

    def test_dft(self):
        '''test dft with fft'''
        a = np.arange(6).reshape((3, 2))
        bas = np.fft.fft2(a)
        ref = dft(a)
        np.testing.assert_almost_equal(bas, ref, decimal=5)

    def test_fft(self):
        a = np.random.randn(4, 3)
        b = np.random.randn(3, 3)

        from scipy import signal
        bas = signal.fftconvolve(a, b, 'same')

        ref = fftconvolve(a, b)
        np.testing.assert_almost_equal(bas, ref, decimal=5)


if __name__ == '__main__':
    unittest.main()