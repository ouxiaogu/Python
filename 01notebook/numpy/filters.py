# coding: utf-8
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

__all__ = ["gaussian_filter", "padding", "convolve", "fftconvolve"
          "nearest_power"]

def gaussian_filter(sigma=2, derivative_order=0):
    """
    Generate Gaussian filter or its derivative with the input sigma
    g(x) = 1/ (sqrt(2*pi)*sigma) * e^(-1/2*x^2/sigma^2)
    g_n_(x) = -1/sigma^2 *( g_(n-1)_(x)*x + g_(n-2)*(x)*(n-1) )
    """
    hlFltSz = int(max(3, math.ceil(3*sigma + 0.5) ))
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

def padding(src, padsize, padval=0):
    """
    padding data with padsize in both directions with given value padval
    """
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
            dstShape = map(lambda x: x + 2*padsize, srcShape)
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
        raise NotImplementedError("convolve only support 1D filter, input shape is %s!\n".format(str(fltShape)) )
    hlFltSz = (array_long_axis_size(flt) - 1)/2
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
        dst = np.asarray(map(lambda i: np.dot(tmp[i:(i + 2*hlFltSz+1)], fltFlipUd), range(arrSz))) # already in src's length
        dst = dst.reshape(srcShape)
    else:
        nrows, ncols = srcShape
        tmp = 1.0*arrPad.copy()
        for row in range(nrows): # apply flt1d in x axis
            tmp[row+hlFltSz, hlFltSz:(ncols+hlFltSz)] = np.asarray(map(lambda i: np.dot(arrPad[row+hlFltSz, i:(i + 2*hlFltSz+1)].flatten(), fltFlipUd), range(ncols) ) )
        dst = tmp.copy()
        for col in range(ncols): # apply flt1d in y axis
            dst[hlFltSz:(nrows+hlFltSz), col+hlFltSz] = np.asarray(map(lambda i: np.dot(tmp[i:(i + 2*hlFltSz+1), col+hlFltSz].flatten(), fltFlipUd), range(nrows) ) )
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
        raise NotImplementedError("convolve only support 1D filter, input shape is %s!\n".format(str(fltShape)) )
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

if __name__ == '__main__':
    # plot_fltG()
    arr = np.random.randn(3, 4)
    fltG = gaussian_filter(1)
    print fftconvolve(arr, fltG)
    print convolve(arr, fltG)
