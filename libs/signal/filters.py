# -*- coding: utf-8 -*-
"""
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

__all__ = ["gaussian_filter", "padding", "padding_backward", "convolve",
            "fftconvolve", "nearest_power", "dft"]

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
        return cv_gaussian_kernel(SMALL_GAUSSIAN_TAB[ksize>>1] )
    if ksize % 2 == 0:
        tmp = ksize + 1
        sys.stderr.write("Warning, kernel size should be odd, adjust ksize from {} to {}".format(ksize, tmp))
        ksize = tmp
    fltSz = ksize
    ksize = float(ksize)
    if(sigma <= 0):
        sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    func_G = lambda i: math.exp(- (i - (ksize-1)/2)**2 / (2*sigma**2) )
    flt_G = np.asarray(list(func_G(i) for i in range(fltSz)) )
    a = np.sum(flt_G)
    return flt_G/a

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

def padding_backward(src, padshape, padval=0):
    arr = np.asarray(src)
    srcShape = arr.shape
    dst = None
    if(is_1D_array(arr) ):
        if(not isinstance(padshape, int)):
            raise ValueError("padshape should be int type!\n")
        arrSz = array_long_axis_size(arr)
        arr = arr.reshape((arrSz,) )
        dst = padval * np.ones(arr, arr.dtype)
        dst[0:arrSz] = arr
        dst = dst.reshape(srcShape)
    elif(len(srcShape) == 2):
        if(len(padshape) != 2):
            raise ValueError("padshape {} should be the same with src {}!\n".format(padshape, srcShape))
        nrows, ncols = srcShape
        newshape = tuple(np.sum(a) for a in zip(srcShape, padshape) )
        dst = padval * np.ones(newshape, arr.dtype)
        dst[0:nrows, 0:ncols] = arr
    else:
        raise NotImplementedError("padding_backward only support 1D/2D array, input array shape is: {}!\n".format(str(srcShape)))
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
    sys.path.append("../common")
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
    sys.path.append("../common")
    from PlotConfig import choosePalette, addLegend
    pal = choosePalette()
    fig, ax = plt.subplots()
    for ix, mode in enumerate(['CV', 'RD', 'OCE']):
        flt_G = gaussian_filter(sigma, mode=mode)
        hlFltSz = (len(flt_G) - 1)//2
        x = list(map(lambda i: i-hlFltSz, range(0, 2*hlFltSz + 1)))
        ax.plot(x, flt_G, '-s', color=pal[ix], label="mode={}, sigma={}, hlFltSz={}".format(mode, sigma, hlFltSz) )
    addLegend([ax ])

def plot_rect_func_fft():
    sys.path.append("../common")
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

def dft_1d_matrix(N):
    '''build 1D dft matrix with W_i = u^i'''
    x, y = np.meshgrid(np.arange(N), np.arange(N))
    v = np.exp( - 2 * np.pi * 1J / N )
    W = np.power(v, x*y) #/ np.sqrt(N)
    return W

def dft_2d_matrix(N, M):
    '''build 2D dft matrix with W_ij = u^i*v^j
    N: #rows, M: #columns,

    Example
    -------
    # x, y = meshgrid(np.arange(M), np.arange(N))
    # Think meshgrid(col, row) as a 2D x-y axis grids, then
    # get x, y coordinates, firstly by column, then by row
    x, y = np.meshgrid(np.arange(3), np.arange(2))
    # => x = { [0, 1, 2], [0, 1, 2] }
    # => y = { [0, 0, 0], [1, 1, 1] }
    '''
    u = np.exp( - 2. * np.pi * 1J / M )
    v = np.exp( - 2. * np.pi * 1J / N )

    x, y = np.meshgrid(np.arange(M), np.arange(M))
    U = np.power(u, x*y) #/ np.sqrt(M)

    x, y = np.meshgrid(np.arange(N), np.arange(N))
    V = np.power(v, x*y) #/ np.sqrt(N)
    return U, V

def dft(src):
    '''
    DFT computation by directly matrix computation, complementary to fft
    Note, unlike DIP book DFT convention, DFT has coefficients 1/sqrt(MN),
    so as inverse DFT also should multiple 1/(MN).

    Steps:
        1. compute by: DFT = V * f(x, y) * U
        2. DFT on Y, F(x, v) = sum(exp(-j2πvy/N), f(x,y)) | y=0->N-1
        See x as known, then Y' = V*Y

        V: size NxN, applied col by col
            [       1,       1,     ..,       1],
        V = [       1,     v^2,     ..,     v^N],
            [      ..,      ..,     ..,      ..],
            [       1,     v^N,     ..,v^(2N-2)]

        3. DFT on X, F(u, v) = sum(exp(-j2πux/M), F(x, v)) | x=0->M-1
        See v as known, then X' = X*U

        U: size MxM, applied row by row
            [       1,       1,     ..,       1],
        U = [       1,     u^2,     ..,     u^M],
            [      ..,      ..,     ..,      ..],
            [       1,     u^M,     ..,u^(2M-2)]

        size: DFT = (V* src * U), should be (NxN) * (N*M) * (M*M)

        4. traceback back,
            - for 2D DFT, we just need to prepare NxN matrix U and MxM matrix V
              DFT = V* src * U
            - for 1D DFT, is just 1 NxN matrix: DFT = V* src

    Parameters
    ----------
    src : array_like
        object to apply DFT, mostly likely is 2D image array, but 1D array
        is also supported

    Returns
    -------
    dst : array_like
        DFT result object, same shape as src

    Reference
    ---------
    https://stackoverflow.com/questions/19739503/dft-matrix-in-python
    '''
    arr = np.asarray(src)
    srcShape = arr.shape
    dst = None
    if(is_1D_array(arr) ):
        N = array_long_axis_size(arr)
        arr = arr.reshape((1,N) )
        W = dft_1d_matrix(N)
        dst = W.dot(arr)
        dst = dst.reshape(srcShape)
    elif(len(srcShape) == 2):
        N, M = srcShape
        U, V = dft_2d_matrix(N, M)
        dst = V.dot(arr).dot(U)
    else:
        raise NotImplementedError("DFT only support 1D/2D array, input array shape is: {}!\n".format(str(srcShape)))
    return dst

def dft_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1J / N )
    W = np.power( omega, i * j ) #/ np.sqrt(N)
    return W

if __name__ == '__main__':
    # arr = np.random.randn(3, 4)
    # fltG = gaussian_filter(1)
    # # print(fftconvolve(arr, fltG) )
    # # print(convolve(arr, fltG) )

    # plot_flt_sz(2)

    # print(gaussian_filter(2))
    # a = cv_gaussian_kernel(11)
    # print(a)
    # amin = np.min(a)
    # print(list(x//amin for x in a))
    # import cv2
    # print(cv2.getGaussianKernel(11, -1))

    # plot_rect_func_fft()

    '''test dft matrix'''
    print(dft_1d_matrix(3))
    U, V = dft_2d_matrix(2, 2)
