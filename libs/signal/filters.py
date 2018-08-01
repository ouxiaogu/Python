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
import collections

__all__ = ['gaussian_filter', 'padding', 'padding_backward', 'convolve',
        'fftconvolve', 'nearest_power', 'dft', 'cv_gaussian_kernel',
        'correlate', 'applySepFilter', 'kernelPreProc',
        'applyKernelOperator', 'centered',
        ]

SMALL_GAUSSIAN_TAB = [
    [1.],
    [0.25, 0.5, 0.25],
    [0.0625, 0.25, 0.375, 0.25, 0.0625],
    [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]
]
SMALL_GAUSSIAN_SIZE = 7

def gaussian_filter(sigma=2, derivative_order=0, mode='CV', dtype=None):
    """
    Generate Gaussian filter or its derivative with the input sigma
    g(x) = 1/ (sqrt(2*pi)*sigma) * e^(-1/2*x^2/sigma^2)
    g_n_(x) = -1/sigma^2 *( g_(n-1)_(x)*x + g_(n-2)*(x)*(n-1) )
    """
    hlFltSz = int(max(2, (sigma-0.8)/0.3 + 1))
    lookuped = False
    if mode == 'CV': # default mode
        ksize = 2*hlFltSz + 1
        if(ksize <= SMALL_GAUSSIAN_SIZE and derivative_order == 0):
            dst = SMALL_GAUSSIAN_TAB[ksize>>1]
            lookuped = True
    elif mode == 'RD':
        hlFltSz = int(max(3, math.ceil(3.5*sigma) ))
    elif mode == 'OCE':
        hlFltSz = int(abs(sigma))
    else:
        raise ValueError("gaussian_filter, only support mode 'CV', 'RD', or 'OCE'!\n")

    if not lookuped:
        if derivative_order == 0:
            a = 1. / (math.sqrt(2*math.pi) * sigma)
            flt_G = np.asarray(list(map(lambda x: a * math.exp(- (x - hlFltSz)**2 / (2.*sigma**2) ), range(0, 2*hlFltSz+1) ) ))
            ss = np.sum(flt_G)
            dst = flt_G/ss
        elif derivative_order == 1:
            fltG0 = gaussian_filter(sigma, 0)
            dst = np.asarray(list(map(lambda x, g0: -1./(sigma**2)*g0*(x - hlFltSz), range(len(fltG0)), fltG0) ) )
        elif derivative_order >=2:
            fltG0 = gaussian_filter(sigma, derivative_order-2)
            fltG1 = gaussian_filter(sigma, derivative_order-1)
            dst = np.asarray(list(map(lambda x, g0, g1: -1./(sigma**2)*(g1*(x - hlFltSz) + g0*(derivative_order - 1)), range(len(fltG0)), fltG0, fltG1) ) )
        else:
            raise NotImplementedError("gaussian_filter don't support negative derivative order: {}!\n".format(derivative_order) )
    if dtype is not None and 'int' in str(dtype):
        vmin = np.min(dst)
        dst = np.floor(dst/vmin + 0.5)
        dst = dst.astype(dtype)
    return dst

def cv_gaussian_kernel(ksize, sigma=0, dtype=None):
    try:
        ksize = int(ksize)
    except:
        raise TypeError("cv_gaussian_kernel is to generate linear Gaussian filter, ksize should be int, but input is: {}!\n".format(str(ksize)))
    if(ksize <= SMALL_GAUSSIAN_SIZE):
        dst = np.array(SMALL_GAUSSIAN_TAB[ksize>>1])
    else:
        if ksize % 2 == 0:
            tmp = ksize + 1
            sys.stderr.write("Warning, kernel size should be odd, adjust ksize from {} to {}!\n".format(ksize, tmp))
            ksize = tmp
        fltSz = ksize
        ksize = float(ksize)
        if(sigma <= 0):
            sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
        func_G = lambda i: math.exp(- (i - (ksize-1)/2)**2 / (2*sigma**2) )
        flt_G = np.asarray(list(func_G(i) for i in range(fltSz)) )
        a = np.sum(flt_G)
        dst = flt_G/a

    if dtype is not None:
        if 'int' in str(dtype):
            vmin = np.min(dst)
            dst = np.floor(dst/vmin + 0.5)
        dst = dst.astype(dtype)
    return dst

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

def padding(src, padshape, padval=0):
    """
    padding data with padsize in both directions with given value padval
    """
    arr = np.asarray(src)
    srcShape = arr.shape
    dst = None
    if(is_1D_array(arr) ):
        if(isinstance(padshape, collections.Iterable)):
            raise ValueError("padshape should be not iterable type, not {}!\n".format(str(type(padshape)) ))
        n = padshape
        arrSz = array_long_axis_size(arr)
        arr = arr.reshape((arrSz,) )
        dst = padval * np.ones(arrSz + 2*n, arr.dtype)
        dst[n:(arrSz+n)] = arr

        dstShape = tuple(a+2*n if a != 1 else a for a in srcShape)
        dst = dst.reshape(dstShape)
    elif(len(srcShape) == 2):
        try:
            if(len(padshape) != 2):
               raise ValueError("padshape's dimension {} should be the same with src's {}!\n".format(padshape, srcShape))
        except:
            padshape = (padshape, padshape)
        n, m = padshape
        dstShape = tuple(s + 2*a for s, a in zip(srcShape, padshape) )
        dst = padval*np.ones(dstShape, arr.dtype)
        dst[n:(n + srcShape[0]), m:(m + srcShape[1])] = arr
    else:
        raise NotImplementedError("padding only support 1D/2D array, input array shape is: {}!\n".format(str(srcShape)))
    return dst

def padding_backward(src, padshape, padval=0):
    arr = np.asarray(src)
    srcShape = arr.shape
    dst = None
    if(is_1D_array(arr) ):
        if(isinstance(padshape, collections.Iterable)):
            raise ValueError("padshape should be not iterable type, not {}!\n".format(str(type(padshape)) ))
        arrSz = array_long_axis_size(arr)
        arr = arr.reshape((arrSz,) )
        dst = padval * np.ones(arrSz+padshape, arr.dtype)
        dst[0:arrSz] = arr
        dst = dst.reshape(srcShape)
    elif(len(srcShape) == 2):
        try:
            if(len(padshape) != 2):
                raise ValueError("padshape {} should be the same with src {}!\n".format(padshape, srcShape))
        except TypeError:
            padshape = (padshape, padshape)
        nrows, ncols = srcShape
        try:
            newshape = tuple(np.sum(a) for a in zip(srcShape, padshape) )
        except TypeError:
            print(srcShape, padshape, sep='\n')
            raise TypeError
        dst = padval * np.ones(newshape, arr.dtype)
        dst[0:nrows, 0:ncols] = arr
    else:
        raise NotImplementedError("padding_backward only support 1D/2D array, input array shape is: {}!\n".format(str(srcShape)))
    return dst

def convolve(src, fltx, flty=None):
    """
    Convolve src with 2 1D-filters, also support convolve 1D array with fltx
    """
    fltx = np.flipud(fltx)
    if flty is not None:
        flty = np.flipud(flty)
    dst = applySepFilter(src, fltx, flty)
    return dst

def applySepFilter(src, fltx, flty=None):
    """
    Convolve src with fltx/flty, by default, padding src with 0

    Parameters
    ----------
    src : array_like
          input object to be convolved
    fltx : 1D array_like
          filter, will convolve into all dimension of src

    Returns
    -------
    dst : array_like
          apply filters result, same shape as src
    """
    arr = np.asarray(src)
    srcShape = arr.shape
    if(len(srcShape) > 2):
        raise NotImplementedError("applySepFilter only support 1D/2D array, input array shape is: {}!\n".format(str(srcShape)))
    if flty is None:
        flty = fltx

    fltx = np.asarray(fltx)
    flty = np.asarray(flty)
    if(np.ndim(fltx) != 1 or np.ndim(flty) != 1):
        raise NotImplementedError("applySepFilter only support 1D filter, input  fltx shape: %s, fltx shape: %s!\n".format(str(fltx.shape), str(flty.shape) ) )
    n, m = array_long_axis_size(flty), array_long_axis_size(fltx)
    if n%2 == 0 or m%2 == 0:
        raise ValueError("applySepFilter only support odd filter size, fltx: {}, flty: {}".format(str(fltx.shape), str(flty.shape)))
    hlFltXSz = array_long_axis_size(fltx)//2
    hlFltYSz = array_long_axis_size(flty)//2

    dst = None
    if is_1D_array(arr):
        arrSz = array_long_axis_size(arr)
        arrPad = padding(arr, hlFltXSz)
        tmp = arrPad.flatten()
        dst = np.asarray(list(map(lambda i: np.dot(tmp[i:(i + 2*hlFltXSz+1)], fltx), range(arrSz)))) # already in src's length
        dst = dst.reshape(srcShape)
    else:
        nrows, ncols = srcShape
        arrPad = padding(arr, (hlFltYSz, hlFltXSz))
        tmp = 1.0*arrPad.copy()
        for row in range(nrows): # apply flt1d in x axis
            tmp[row+hlFltYSz, hlFltXSz:(ncols+hlFltXSz)] = np.asarray(list(map(lambda i: np.dot(arrPad[row+hlFltYSz, i:(i + 2*hlFltXSz+1)].flatten(), fltx), range(ncols) ) ) )
        dst = tmp.copy()
        for col in range(ncols): # apply flt1d in y axis
            dst[hlFltYSz:(nrows+hlFltYSz), col+hlFltXSz] = np.asarray(list(map(lambda i: np.dot(tmp[i:(i + 2*hlFltYSz+1), col+hlFltXSz].flatten(), flty), range(nrows) ) ) )
        dst = dst[hlFltYSz:(hlFltYSz+nrows), hlFltXSz:(hlFltXSz+ncols)]
    return dst

def correlate(src, fltx, flty=None):
    '''
    Flow of correlation is similar with convolve, just don't need to flip
    '''
    dst = applySepFilter(src, fltx, flty)
    return dst


def is_1D_array(src):
    """len(shape)==1 or min(shape)=1"""
    arr = np.asarray(src)
    srcShape = arr.shape
    if(len(srcShape) == 1 or np.min(srcShape)== 1):
        return True
    return False

def array_long_axis_size(src):
    """get array's size at longest axis"""
    return max(src.shape)

def centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    curshape = np.asarray(arr.shape)
    startind = (curshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def fftconvolve(src, flt, mode='same'):
    """
    What different from spatial convolve is:
        1. use a nearest shape as 2^i*3^j*5^k to compute fft will save runtime,
           here just pad into the nearest 2^n
        2. convolve 2D input with 1D filter, we can compose a 2D filter first,
           because of the separability of 2D convolution
           f(x,y)*g(x,y) = f(x,y)*(g(x).g(y)) = f(x,y)*g(x)*g(y)
           http://www.songho.ca/dsp/convolution/convolution2d_separable.html
        3. minimum shape: shape = s1 + s2 - 1, then mapping to nearest_power
            for fft accelerate
    """
    arr = np.asarray(src)
    srcShape = arr.shape
    flt = np.asarray(flt)
    fltSz = array_long_axis_size(flt)

    if is_1D_array(arr):
        fltShape = ( 1 if s==1 else fltSz for s in srcShape)
        fltRes = flt.reshape(fltShape )
    elif np.ndim(arr) == 2:
        if np.ndim(flt) == 1:
            fltRes = flt.reshape((fltSz, 1) ) * flt.reshape((1, fltSz) )
        else:
            fltRes = flt
    elif(arr.ndim > 2):
        raise NotImplementedError("convolve only support 1D/2D array, input array shape is: {}!\n".format(str(srcShape)))

    if arr.ndim != fltRes.ndim:
        raise ValueError("array and filter should have the same dimensionality")

    s1 = np.asarray(arr.shape)
    s2 = np.asarray(fltRes.shape)
    if(mode == "valid" and np.any(s1 - s2 < 0)):
        raise NotImplementedError("arr size should be larger than filter size in 'valid' mode")
    shape = s1 + s2 - 1
    fshape = [nearest_power(d) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape]) # backward padding
    sp1 = np.fft.rfftn(arr, fshape)
    sp2 = np.fft.rfftn(fltRes, fshape)
    ret = np.fft.irfftn(sp1 * sp2, fshape)[fslice].copy() # backward padding

    if mode == "same":
        return centered(ret, s1)
    elif mode == "valid":
        return centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid' or 'same'.")
    return ret

def nearest_power(num, powbase=2):
    num = float(num)
    if(num <= powbase):
        return powbase
    return powbase*nearest_power(num/powbase)

def plot_flt_dG(sigma=2):
    sys.path.append((os.path.dirname(os.path.abspath(__file__)))+"/../common")
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
    sys.path.append((os.path.dirname(os.path.abspath(__file__)))+"/../common")
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
    sys.path.append((os.path.dirname(os.path.abspath(__file__)))+"/../common")
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

def kernelPreProc(src, ksize=None):
    '''
    Parameters
    ----------
    src : 2D array-like
        padded image array
    ksize : int or tuple
        kernel size

    Returns
    -------
    dst : 2D array-like
        dst, as zeros_like of the src
    gp : 2D array-like
        padded src image array
    N, M : int
        image #rows, #columns
    n, m : int
        kernel #rows, #columns, both should be odd
    hlFltSzY, hlFltSzX (cy, cx):
        half kernel size in rows(Y) and columns(X), also the center x/y
        of the kernel, named as (cy, cx)
    '''
    if ksize is None:
        ksize = (3, 3)
    elif np.ndim(ksize) == 0:
        ksize = (ksize, ksize)
    N, M = src.shape
    n, m = ksize
    if n%2 == 0 or m%2 == 0:
        raise ValueError("ksize shape {} should be odd!\n".format(repr(ksize)))
    hlFltSzY, hlFltSzX = n//2, m//2
    gp = padding(src, (hlFltSzY, hlFltSzX) )
    dst = np.zeros_like(src)
    return dst, gp, N, M, n, m, hlFltSzY, hlFltSzX

def fltGenPreProc(shape=None):
    '''
    N, M : int
        kernel #rows, #columns, both should be odd
    cy, cx :
        half kernel size in rows(Y) and columns(X)
    '''
    if shape is None:
        shape = (3, 3)
    elif np.ndim(shape) == 0:
        shape = (shape, shape)
    N, M = shape
    if N%2 == 0 or M%2 == 0:
        raise ValueError("filter shape size {} should be odd!\n".format(repr(shape)))
    cy, cx = (s//2 for s in shape)
    return N, M, cy, cx

def applyKernelOperator(src, ksize, operator=None):
    '''
    apply kernel level operator, return one value from the kernel pixels

    operators like:
        - statistics: min, max, mean, std
    '''
    dst, fp, N, M, n, m, _, _ = kernelPreProc(src, ksize)
    if(isinstance(operator, collections.Iterable)):
        dst = tuple(dst for i in range(len(operator)))
    for r in range(N):
        for c in range(M):
            slices = fp[r:(r+n), c:(c+m)]
            if(isinstance(operator, collections.Iterable)):
                for i, opt in enumerate(operator):
                    dst[i][r, c] = opt(slices)
            else:
                dst[r, c] = operator(slices)
    return dst

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
