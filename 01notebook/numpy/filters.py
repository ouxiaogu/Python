# coding: utf-8
import math
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(r'C:\Users\peyang\Perforce\peyang_LT324319_3720\app\mxp\scripts\util')

import logger
logger.initlogging(debug=False)
log = logger.getLogger("filters")

__all__ = ["gaussian_filter", "padding", "convolve"]

def gaussian_filter(sigma=2, derivative_order=0):
    """
    Generate Gaussian filter or its derivative with the input sigma
    g(x) = 1/ (sqrt(2*pi)*sigma) * e^(-1/2*x^2/sigma^2)
    g_n_(x) = -1/sigma^2 *( g_(n-1)_(x)*x + g_(n-2)*(x)*(n-1) )
    """
    hlFltSz = int(max(3, math.ceil(3*sigma + 1) ))
    if derivative_order == 0:
        a = 1. / (math.sqrt(2*math.pi) * sigma)
        return map(lambda x: a * math.exp(-(x - hlFltSz) * (x - hlFltSz)/ (2.*sigma*sigma) ), range(0, 2*hlFltSz+1) )
    elif derivative_order == 1:
        fltG0 = gaussian_filter(sigma, 0)
        return map(lambda x, g0: -1./(sigma*sigma)*g0*(x - hlFltSz), range(len(fltG0)), fltG0)
    elif derivative_order >=2:
        fltG0 = gaussian_filter(sigma, derivative_order-2)
        fltG1 = gaussian_filter(sigma, derivative_order-1)
        return map(lambda x, g0, g1: -1./(sigma*sigma)*(g1*(x - hlFltSz) + g0*(derivative_order - 1)), range(len(fltG0)), fltG0, fltG1)
    else:
        raise NotImplementedError("gaussian_filter don't support negative derivative order: {}!\n".format(derivative_order) )

def padding(inarray, padsize, padval=0):
    """
    padding data with padsize in both directions with given value padval
    """
    arr = np.asarray(inarray)
    inshape = arr.shape
    out = None
    if len(inshape) == 1:
        outshape = (inshape[0]+2*padsize, )
        out = padval*np.ones(outshape)
        out[padsize:(padsize + inshape[0])] = arr
    elif len(inshape) == 2:
        if inshape[1] == 1:
            outshape = (inshape[0]+2*padsize, 1)
            out = padval*np.ones(outshape)
            out[padsize:(padsize + inshape[0]), 0] = arr.flatten()
        elif inshape[0] == 1:
            outshape = (1, inshape[1]+2*padsize)
            out = padval*np.ones(outshape)
            out[0, padsize:(padsize + inshape[0])] = arr.flatten()
        else:
            outshape = map(lambda x: x + 2*padsize, inshape)
            out = padval*np.ones(outshape)
            out[padsize:(padsize + inshape[0]), padsize:(padsize + inshape[1])] = arr
    else:
        raise NotImplementedError("padding don't support array with such shape: {}!\n".format(str(inshape)))
    return out

def convolve(inarray, flt, padding=False):
    arr = np.asarray(inarray)
    hlFltSz = (len(flt) - 1)/2
    ouarr = padding(arr, hlFltSz)


def plot_fltG():
    from PlotConfig import choosePalette, addLegend
    pal = choosePalette()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    sigma = 2
    fltG = gaussian_filter(sigma)
    hlFltSz = (len(fltG) - 1)/2
    x = map(lambda i: i-hlFltSz, range(0, 2*hlFltSz + 1))
    for i in range(0, 3):
        fltGG = gaussian_filter(sigma, i)
        ax.plot(x, fltGG, '-s', color=pal[i], label="sigma=%d, order=%d" % (sigma, i))

    addLegend([ax])

if __name__ == '__main__':
    plot_fltG()

