import math
import unittest
import numpy as np
import matplotlib.pyplot as plt
import scipy

import sys
import os.path
sys.path.insert(0, os.path.abspath("../"))
from filters import *

sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../common")
from PlotConfig import choosePalette, addLegend

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

def plot_flt_dG(sigma=2):
    sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../common")
    from PlotConfig import choosePalette, addLegend
    pal = choosePalette()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    fltG = gaussian_filter(sigma)
    hlFltSz = (len(fltG) - 1)//2
    x0 = list(map(lambda i: i-hlFltSz, range(0, 2*hlFltSz + 1)))
    for ix in range(0, 3):
        fltGG = gaussian_filter(sigma, ix)
        fltGG = np.asarray(fltGG)
        axes[0].plot(x0, fltGG, '-', color=pal[ix], label="sigma=%f, order=%d" % (sigma, ix))
        sp = np.fft.rfftn(fltGG, fltGG.shape)
        sp = np.fft.fftshift(sp)
        x1 = range(len(sp))
        mag = np.absolute(sp)
        axes[1].plot(x1, mag, '-*', color=pal[ix], label="sigma=%f, order=%d, spectrum mag" % (sigma, ix))
    addLegend([axes[0]])
    addLegend([axes[1]])

def plot_flt_sz(sigma=2):
    pal = choosePalette()
    fig, ax = plt.subplots()
    for ix, mode in enumerate(['CV', 'RD', 'OCE']):
        flt_G = gaussian_filter(sigma, mode=mode)
        hlFltSz = (len(flt_G) - 1)//2
        x = list(map(lambda i: i-hlFltSz, range(0, 2*hlFltSz + 1)))
        ax.plot(x, flt_G, '-s', color=pal[ix], label="mode={}, sigma={}, hlFltSz={}".format(mode, sigma, hlFltSz) )
    addLegend([ax ])

def plot_rect_func_fft():
    sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../common")
    from MathUtil import rectFunc
    fig, axes = plt.subplots(nrows=3, ncols=1)

    x, y = zip(*rectFunc(2, slices=20, extend=1) )
    axes[0].plot(x, y, 'b-s', label='rectangle function')

    sp1 = np.fft.fft(y)
    sp1 = np.fft.fftshift(sp1)
    axes[1].plot(x, sp1, 'g-*', label='rectangle function, fft')

    sp2 = np.fft.rfft(y)
    # sp2 = np.fft.fftshift(sp2)
    print('length of sp1: {}, sp2: {}'.format(len(sp1), len(sp2)) )
    axes[2].plot(np.arange(len(sp2)), sp2, 'k-.', label='rectangle function, fft for real input')
    from PlotConfig import addLegend
    addLegend(axes)
    plt.show()

def plot_DoG():
    fig, axes = plt.subplots(nrows=2, ncols=2)
    print(type(axes), type(axes[0,1]))

    sigma1 = 0.3
    sigma2 = sigma1/1.67
    ksize = 101
    cx = ksize//2
    flt_G1 = cv_gaussian_kernel(ksize, sigma1)
    cir_G1 = np.roll(flt_G1, -cx)

    flt_G2 = cv_gaussian_kernel(ksize, sigma2)

    factor = np.sqrt(2*np.pi)
    DoG = 2*factor*sigma1*flt_G1 - factor*sigma2*flt_G2
    cir_DoG = np.roll(DoG, -cx)

    x = np.arange(-50, 51)

    sp = idft(cir_DoG)
    np_sp_shift = np.fft.fftshift(np.fft.ifft(cir_DoG))
    sp_shift = np.zeros_like(sp)
    sp_shift[:50] = sp[51:].copy()
    sp_shift[50:] = sp[:51].copy()
    sp_shift_amp = np.real(sp_shift)
    print(np_sp_shift[0:101:10])
    print(sp_shift[0:101:10])

    sp1 = np.fft.fftshift(np.fft.ifft(factor*sigma1*cir_G1))
    print(sp1[0:101:10])
    sp1_amp = np.real(sp1)

    axes[0,0].plot(x, factor*sigma1*flt_G1, label='GLPF1, $\sigma={:.2f}$'.format(sigma1))

    axes[0,1].plot(x, 1/np.max(flt_G1)*flt_G1, label='GLPF1, $\sigma={:.2f}$'.format(sigma1))
    axes[0,1].plot(x, 1/np.max(flt_G2)*flt_G2, label='GLPF2, $\sigma={:.2f}$'.format(sigma2))
    axes[0,1].plot(x, DoG, label='Frequency DoG')

    sigma1_s = np.std(sp1_amp)
    from scipy.stats import norm
    mu, std = norm.fit(factor*sigma1*flt_G1)
    print(mu, std)
    mu, std = norm.fit(sp1_amp)
    print(mu, std)
    axes[1,0].plot(x, sp1_amp, label='GLPF1 iDFT, $\sigma={:.3e}$'.format(sigma1_s))

    axes[1,1].plot(x, sp_shift_amp, label='Spatial DoG')
    for i in range(2):
        for j in range(2):
            addLegend([axes[i, j]])
    plt.show()

def plot_fft_G():
    # even function in discrete is DIP 151, not the same with continious function
    fig, axes = plt.subplots(nrows=4, ncols=1)

    sigma1 = 0.80
    ksize = 101
    cx = ksize//2
    flt_G1 = cv_gaussian_kernel(ksize, sigma1)
    cir_G1 = np.roll(flt_G1, -cx)
    print("isEven flt_G1", isEven(flt_G1))
    print("isEven cir_G1", isEven(cir_G1))

    x = np.arange(-50, 51)
    factor = np.sqrt(2*np.pi)

    sp1 = np.fft.ifft(cir_G1)
    print(sp1[0:100:10])

    axes[0].plot(x, np.fft.fftshift(factor*sigma1*cir_G1), label='flt_G1, $\sigma={:.2f}$'.format(sigma1))

    axes[1].plot(x, np.fft.fftshift(np.real(sp1)), label='flt_G1 DFT')

    # even test1, add one element before the list
    a1 = [flt_G1[cx] ] + flt_G1.tolist()
    ax = np.arange(-51, 51)
    print("isEven a1", isEven(a1))


    # even test2, move the position of one element: a[cx] into a[0]
    a2 = np.zeros(ksize )
    a2[0] = flt_G1[cx]
    a2[1:(cx+1)] = flt_G1[:cx]
    a2[(cx+1):] = flt_G1[(cx+1):]
    print("isEven a2", isEven(a2))
    axes[2].plot(x, factor*sigma1*a2, label='a2, $\sigma={:.2f}$'.format(sigma1))
    sp_a2 = np.fft.ifft(a2)
    print(sp_a2[0:100:10])

    axes[3].plot(x, np.real(sp_a2), label='a2 DFT')

    for i in range(4):
        addLegend([axes[i]])
    plt.show()

def main():
    plot_flt_dG(sigma=8)

    # plot_flt_sz()

    # plot_rect_func_fft()

    #plot_DoG()
    # plot_fft_G()


if __name__ == '__main__':
    # unittest.main()

    main()