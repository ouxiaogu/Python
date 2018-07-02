"""
-*- coding: utf-8 -*-
Created: ouxiaogu, 2018-06-28 15:00:06

Underlying core functions to
  - Build filter
  - Apply filter

Last Modified by: ouxiaogu
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import sys

__all__ = ["gaussian_filter", "padding", "convolve", "fftconvolve"
          "nearest_power"]

SMALL_GAUSSIAN_TAB = [
    [1.],
    [0.25, 0.5, 0.25],
    [0.0625, 0.25, 0.375, 0.25, 0.0625],
    [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]
]
SMALL_GAUSSIAN_SIZE = 7

def gaussian_filter(sigma=2, derivative_order=0, mode='CV'):
    """
    Generate Gaussian filter or its derivative with the input sigma
    g(x) = 1/ (sqrt(2*pi)*sigma) * e^(-1/2*x^2/sigma^2)
    g_n_(x) = -1/sigma^2 *( g_(n-1)_(x)*x + g_(n-2)*(x)*(n-1) )
    """
    hlFltSz = int(max(2, (sigma-0.8)/0.3 + 1))
    if mode == 'CV': # default mode
        ksize = 2*hlFltSz + 1
        if(ksize <= SMALL_GAUSSIAN_SIZE and derivative_order == 0):
            return SMALL_GAUSSIAN_TAB[ksize>>1]
    elif mode == 'RD':
        hlFltSz = int(max(3, math.ceil(3.5*sigma) ))
    elif mode == 'OCE':
        hlFltSz = int(abs(sigma))
    else:
        raise ValueError("gaussian_filter, only support mode 'CV', 'RD', or 'OCE'!\n")

    if derivative_order == 0:
        a = 1. / (math.sqrt(2*math.pi) * sigma)
        flt_G = np.asarray(list(map(lambda x: a * math.exp(- (x - hlFltSz)**2 / (2.*sigma**2) ), range(0, 2*hlFltSz+1) ) ))
        ss = np.sum(flt_G)
        return flt_G/ss
    elif derivative_order == 1:
        fltG0 = gaussian_filter(sigma, 0)
        return np.asarray(list(map(lambda x, g0: -1./(sigma**2)*g0*(x - hlFltSz), range(len(fltG0)), fltG0) ) )
    elif derivative_order >=2:
        fltG0 = gaussian_filter(sigma, derivative_order-2)
        fltG1 = gaussian_filter(sigma, derivative_order-1)
        return np.asarray(list(map(lambda x, g0, g1: -1./(sigma**2)*(g1*(x - hlFltSz) + g0*(derivative_order - 1)), range(len(fltG0)), fltG0, fltG1) ) )
    else:
        raise NotImplementedError("gaussian_filter don't support negative derivative order: {}!\n".format(derivative_order) )

def cv_gaussian_kernel(ksize, sigma=0):
    ksize = int(ksize)
    if(ksize <= SMALL_GAUSSIAN_SIZE):
        return SMALL_GAUSSIAN_TAB[ksize>>1]
    if ksize % 2 == 0:
        tmp = ksize + 1
        sys.stderr("Warning, kernel size should be odd, adjust ksize from {} to {}".format(ksize, tmp))
        ksize = tmp
    fltSz = ksize
    ksize = float(ksize)
    if(sigma <= 0):
        sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    func_G = lambda i: math.exp(- (i - (ksize-1)/2)**2 / (2*sigma**2) )
    flt_G = list(func_G(i) for i in range(fltSz))
    a = np.sum(flt_G)
    flt_G = list(x/a for x in flt_G)
    return flt_G

def gabor_filter(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

def padding(src, padsize, padval=0):
    """
    padding data with padsize in both directions with given value padval
    """
    if(not isinstance(padsize, int)):
        raise ValueError("padsize should be int type!\n")
    arr = np.asarray(src)
    srcShape = arr.shape
    dst = None
    if len(srcShape) == 1:
        dstShape = (srcShape[0]+2*padsize, )
        dst = padval*np.ones(dstShape)
        dst[padsize:(padsize + srcShape[0])] = arr
    elif len(srcShape) == 2:
        if srcShape[1] == 1:
            dstShape = (srcShape[0]+2*padsize, 1)
            dst = padval*np.ones(dstShape)
            dst[padsize:(padsize + srcShape[0]), 0] = arr.flatten()
        elif srcShape[0] == 1:
            dstShape = (1, srcShape[1]+2*padsize)
            dst = padval*np.ones(dstShape)
            dst[0, padsize:(padsize + srcShape[1])] = arr
        else:
            dstShape = list(map(lambda x: x + 2*padsize, srcShape) )
            dst = padval*np.ones(dstShape)
            dst[padsize:(padsize + srcShape[0]), padsize:(padsize + srcShape[1])] = arr
    else:
        raise NotImplementedError("padding only support 1D/2D array, input array shape is: {}!\n".format(str(srcShape)))
    return dst

def convolve(src, flt1d, wipadding=True):
    """
    Convolve src with flt1d, by default, padding src with 0

    Parameters
    ----------
    src : array_like
          input object to be convolved
    flt1d : 1D array_like
          filter, will convolve into all dimension of src
    wipadding : bool
          whether to pad the input `src`, default is True

    Returns
    -------
    dst : array_like
          convolution result, same shape as src
    """
    arr = np.asarray(src)
    srcShape = arr.shape
    if(len(srcShape) > 2):
        raise NotImplementedError("convolve only support 1D/2D array, input array shape is: {}!\n".format(str(srcShape)))

    flt = np.asarray(flt1d)
    if(not is_1D_array(flt) ):
        raise NotImplementedError("convolve only support 1D filter, input shape is %s!\n".format(str(flt.shape)) )
    hlFltSz = (array_long_axis_size(flt) - 1)//2
    flt = flt.flatten() # as 1D flt
    fltFlipUd = np.flipud(flt)

    arrPad = None
    if wipadding:
        arrPad = padding(arr, hlFltSz)
    else:
        arrPad = arr.copy()

    if( (is_1D_array(arrPad) and max(arrPad.shape) < len(flt) ) or
        (not is_1D_array(arrPad) and min(arrPad.shape) < len(flt) ) ):
        raise NotImplementedError("convolve, array or padded array size {} should larger than filter size {}!\n".format(str(arrPad.shape), str(flt.shape)))

    dst = None
    if is_1D_array(arr):
        arrSz = array_long_axis_size(arr)
        tmp = arrPad.flatten()
        dst = np.asarray(list(map(lambda i: np.dot(tmp[i:(i + 2*hlFltSz+1)], fltFlipUd), range(arrSz)))) # already in src's length
        dst = dst.reshape(srcShape)
    else:
        nrows, ncols = srcShape
        tmp = 1.0*arrPad.copy()
        for row in range(nrows): # apply flt1d in x axis
            tmp[row+hlFltSz, hlFltSz:(ncols+hlFltSz)] = np.asarray(list(map(lambda i: np.dot(arrPad[row+hlFltSz, i:(i + 2*hlFltSz+1)].flatten(), fltFlipUd), range(ncols) ) ) )
        dst = tmp.copy()
        for col in range(ncols): # apply flt1d in y axis
            dst[hlFltSz:(nrows+hlFltSz), col+hlFltSz] = np.asarray(list(map(lambda i: np.dot(tmp[i:(i + 2*hlFltSz+1), col+hlFltSz].flatten(), fltFlipUd), range(nrows) ) ) )
        dst = dst[hlFltSz:(hlFltSz+nrows), hlFltSz:(hlFltSz+ncols)]
    return dst

def is_1D_array(src):
    """len(shape)==1 or min(shape)=1"""
    arr = np.asarray(src)
    srcShape = arr.shape
    if(len(srcShape) == 1 or min(srcShape)== 1):
        return True
    return False

def array_long_axis_size(src):
    """get array's size at longest axis"""
    return max(src.shape)

def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    curshape = np.asarray(arr.shape)
    startind = (curshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def fftconvolve(src, flt1d, mode='same'):
    """
    What different from spatial convolve is:
        1. use a nearest shape as 2^i*3^j*5^k to compute fft will save runtime,
           here just pad into the nearest 2^n
        2. convolve 2D input with 1D filter, we can compose a 2D filter first,
           because of the separability of 2D convolution
           f(x,y)*g(x,y) = f(x,y)*(g(x).g(y)) = f(x,y)*g(x)*g(y)
           http://www.songho.ca/dsp/convolution/convolution2d_separable.html
    """
    arr = np.asarray(src)
    srcShape = arr.shape
    if(len(srcShape) > 2):
        raise NotImplementedError("convolve only support 1D/2D array, input array shape is: {}!\n".format(str(srcShape)))

    flt = np.asarray(flt1d)
    if(not is_1D_array(flt) ):
        raise NotImplementedError("convolve only support 1D filter, input shape is %s!\n".format(str(flt.shape)) )
    fltSz = array_long_axis_size(flt)

    if is_1D_array(arr):
        fltRes = flt.reshape(srcShape)
    else:
        fltRes = flt.reshape((fltSz, 1) ) * flt.reshape((1, fltSz) )

    if not arr.ndim == fltRes.ndim:
        raise ValueError("array and filter should have the same dimensionality")

    s1 = np.asarray(arr.shape)
    s2 = np.asarray(fltRes.shape)
    if(mode == "valid" and np.any(s1 - s2 < 0)):
        raise NotImplementedError("arr size should be larger than filter size in 'valid' mode")
    shape = s1 + s2 - 1
    fshape = [nearest_power(d) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    sp1 = np.fft.rfftn(arr, fshape)
    sp2 = np.fft.rfftn(fltRes, fshape)
    ret = (np.fft.irfftn(sp1 * sp2, fshape)[fslice].copy())

    if mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        return _centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid' or 'same'.")
    return ret

def nearest_power(num, powbase=2):
    num = float(num)
    if(num <= powbase):
        return powbase
    return powbase*nearest_power(num/powbase)

def plot_flt_dG(sigma=2):
    import sys
    sys.path.append("../python/util")
    from PlotConfig import choosePalette, addLegend
    pal = choosePalette()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    fltG = gaussian_filter(sigma)
    hlFltSz = (len(fltG) - 1)//2
    x0 = list(map(lambda i: i-hlFltSz, range(0, 2*hlFltSz + 1)))
    for ix in range(0, 3):
        fltGG = gaussian_filter(sigma, ix)
        fltGG = np.asarray(fltGG)
        axes[0].plot(x0, fltGG, '-s', color=pal[ix], label="sigma=%d, order=%d" % (sigma, ix))
        sp = np.fft.rfftn(fltGG, fltGG.shape)
        sp = np.fft.fftshift(sp)
        x1 = range(len(sp))
        mag = np.absolute(sp)
        axes[1].plot(x1, mag, '-*', color=pal[ix], label="sigma=%d, order=%d, spectrum mag" % (sigma, ix))
    addLegend([axes[0]])
    addLegend([axes[1]])

def plot_flt_sz(sigma=2):
    import sys
    sys.path.append("../python/util")
    from PlotConfig import choosePalette, addLegend
    pal = choosePalette()
    fig, ax = plt.subplots()
    for ix, mode in enumerate(['CV', 'RD', 'OCE']):
        flt_G = gaussian_filter(sigma, mode=mode)
        hlFltSz = (len(flt_G) - 1)//2
        x = list(map(lambda i: i-hlFltSz, range(0, 2*hlFltSz + 1)))
        ax.plot(x, flt_G, '-s', color=pal[ix], label="mode={}, sigma={}, hlFltSz={}".format(mode, sigma, hlFltSz) )
    addLegend([ax ])

if __name__ == '__main__':
    # arr = np.random.randn(3, 4)
    # fltG = gaussian_filter(1)
    # # print(fftconvolve(arr, fltG) )
    # # print(convolve(arr, fltG) )

    plot_flt_sz(2)

    # print(gaussian_filter(2))
    # a = cv_gaussian_kernel(11)
    # print(a)
    # amin = np.min(a)
    # print(list(x//amin for x in a))
    # import cv2
    # print(cv2.getGaussianKernel(11, -1))
