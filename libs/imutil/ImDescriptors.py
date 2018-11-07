# -*- coding: utf-8 -*-
'''
Created: peyang, 2018-01-15 16:07:20

ImDescriptors: Image Descriptors Module

Last Modified by:  ouxiaogu
'''

import numpy as np
import cv2
import pandas as pd
from collections import OrderedDict

# mutual import error if add gradient into ImageDescriptors
# from SpatialFlt import SobelFilter

import sys
import os.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../signal")
from filters import cv_gaussian_kernel, applySepFilter, fftconvolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
from PlotConfig import getRGBColor
from logger import logger
log = logger.getLogger(__name__)

__all__ = [ 'RMS_BIN_RANGES', 'ZNCC_BIN_RANGES',
            'printImageInfo', 'getImageInfo',
            'hist_curve', 'hist_rect', 'hist_lines',
            'calcHistSeries', 'calcHist', 'cdfHisto',
            'addOrdereddDict', 'subOrdereddDict', 'Histogram',
            'im_fft_amplitude_phase', 'power_ratio_in_cutoff_frequency',
             'calculate_cutoff',
            'statHist'
        ]

BINS = np.arange(256).reshape(256,1)
RMS_BIN_RANGES = [0, 2, 4, 6, 8, 10, 15, 20, 30, 50, 100]
ZNCC_BIN_RANGES = [0, 0.2, 0.5, 0.8]

def getImageInfo(im):
    shape = 'x'.join(map(str, im.shape))
    return ', '.join(map(str, [shape, im.dtype, np.percentile(im, np.linspace(0, 100, 6),  interpolation='nearest')]))

def printImageInfo(im):
    print(getImageInfo(im))

def hist_curve(im):
    '''return histogram of an image drawn as curves'''
    h = np.zeros((256,256,3))
    if len(im.shape) == 2:
        h = np.zeros((256,256))
        colors = [(255,255)]
    elif im.shape[2] == 3:
        colors = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, colr in enumerate(colors):
        hist_item = cv2.calcHist([im],[ch],None,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((BINS,hist)))
        cv2.polylines(h,[pts],False, colr)
    h=np.flipud(h)
    return h

def hist_lines(im):
    '''return histogram of an image drawn as BINS ( only for grayscale images )
    hist_item is normalized into [0, 255]
    the hist height is set as 300 as default, a little higher than max->255
    '''

    # h = np.zeros((300,256,3), dtype=np.uint8)
    h = np.zeros((256,256), dtype=np.uint8) # output histogram in grayscale
    if len(im.shape)!=2:
        sys.stderr.write("hist_lines applicable only for grayscale images!\n")
        #print("so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)

    hist=np.uint8(np.around(hist_item)) # normalized hist as CV_U8
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255)) # (255,255,255)
    h = np.flipud(h)
    return h

def hist_rect(im=None, hbins=100, hist_in=None, color_hist=True, fit_hist=False):
    '''return histogram of any image as some rectangle bins
    In python, it seems cv2 don't support the float type image,
    But for c++, it's supported

    Parameters
    ----------
    hist_in : None or 1D list-like
        if hist is None, use OpenCV to calcHist
        if hist is 1D list, it should be grayscale histogram, list or dict
    '''
    if hist_in is None:
        if len(im.shape)!=2:
            sys.stderr.write("hist_rect applicable only for grayscale images!\n")
            #print("so converting image to grayscale for representation"
            im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        hist_item = cv2.calcHist([im], [0], None, [hbins], [0, 256])
    else:
        hist_item = hist_in.copy()
    hbins = len(hist_item)
    maxhist = np.max(hist_item)
    hist_item = hist_item/maxhist*255
    # cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
    hist = np.uint8(np.around(hist_item)) # normalized hist as CV_U8

    assert(hbins <= 256)
    height = 260
    scale = height//hbins #hist image size 260 x 256
    if not color_hist:
        histImg = np.zeros((height, 256), np.uint8)
        color = (255,255)
    else:
        histImg = np.zeros((height, 256, 3), np.uint8)
        color = getRGBColor(ix=0) # ix=-3
        bkcolor = getRGBColor(ix=0, background=True)
        histImg[:] = bkcolor
    for ix in range(hbins):
        binVal = hist[ix]
        cv2.rectangle(histImg, (ix*scale, 0), ((ix+1)*scale, binVal), color)

    if fit_hist: # add fitting curve
        # raw & normalized hist, mu & std will have some numerical difference
        # mu, std, _ = statHist(hist_in, trimmedNum=2) # raw hist
        mu, std, yscale = statHist(hist, trimmedNum=2, compute_scale=True) # normalized hist
        normfunc = lambda x: yscale/(np.sqrt(2*np.pi)*std) * np.exp( - (x-mu)**2/(2*(std)**2 ))
        pts = np.int32(np.column_stack((BINS, normfunc(BINS))))
        cv2.polylines(histImg, [pts], False, (0, 0, 0), 1)
    histImg = np.flipud(histImg).copy()
    if fit_hist: # add fitting curve text info
        text = 'u={:.2f}, sigma={:.2f}'.format(mu, std)
        printImageInfo(histImg)
        cv2.putText(histImg, text, (100, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
    return histImg

class Histogram(object):
    """docstring for Histogram"""
    def __init__(self, hist, **kwargs):
        super(Histogram, self).__init__()
        self.hist = hist

    def _validate_args(self, rhs):
        if not isinstance(rhs, Histogram):
            raise TypeError("rhs {} is not the Histogram object!\n".format(type(rhs) ))
        if type(self.hist) != type(rhs.hist):
            raise TypeError("Histogram self {} and rhs {} is not in the same type!\n".format(type(self.hist), type(rhs.hist) ))

    def __add__(self, rhs):
        self._validate_args(rhs)
        if isinstance(self.hist, OrderedDict):
            hist = addOrdereddDict(self.hist, rhs.hist)
        elif isinstance(self.hist, list) or isinstance(self.hist, np.ndarray):
            hist = (np.array(self.hist) + np.array(rhs.hist)).tolist()
        return Histogram(hist)

    def __sub__(self, rhs):
        self._validate_args(rhs)
        if isinstance(self.hist, OrderedDict):
            hist = subOrdereddDict(self.hist, rhs.hist)
        elif isinstance(self.hist, list) or isinstance(self.hist, np.ndarray):
            hist = (np.array(self.hist) - np.array(rhs.hist)).tolist()
        return Histogram(hist)

    def cdf(self):
        return cdfHisto(self.hist)


def fullBinHist(hist):
    '''if hist is OrderedDict type, fill histogram into a full bin histogram version'''
    dst = np.zeros(256, dtype=np.uint32)
    if isinstance(hist, OrderedDict):
        for i in range(256):
            if i in hist.keys():
                dst[i] = hist[i]
    elif isinstance(hist, list) or isinstance(hist, np.ndarray):
        dst = hist.copy()
    else:
        raise TypeError("fullBinHist, hist type {} is not supported".format(type(type(hist))))
    return dst

def statHist(hist=None, trimmedNum=None, compute_scale=False):
    """
    statistics based on the histogram

    Parameters
    ----------
    hist : list like or OrderedDict
        histogram itself
    trimmedNum : int
        trim hist number at the two end by trimmedNum

    Returns
    -------
    mu : float
        mean of the data, computed by sum(bins*pdf)
    std : float
        standard deviation of the data, computed by sqrt(sum((bins^2)*pdf) - mean^2)
    scale : float
        how much the data is scaled, hist point (u, hist_norm[u])
        becomes (u, hist[u]), refer to hist_rect, max(hist)=255,
        but hist[u] not always 255, because trimmed some number
        at head/tail
    """
    hist = fullBinHist(hist)
    #assert(len(hist) == 256)
    if trimmedNum is not None and trimmedNum > 0 and trimmedNum < 255//2:
        hist[:trimmedNum] = 0
        hist[-trimmedNum:] = 0

    hist_norm = hist/np.sum(hist)
    bins = np.arange(len(hist))
    mu = np.dot(bins, hist_norm)
    std = np.sqrt(np.abs(np.dot(bins**2, hist_norm) - mu**2))

    # for compute scale
    scale = None
    if compute_scale:
        flt_G = cv_gaussian_kernel(7)
        hist_smooth = applySepFilter(hist, flt_G)
        hist_norm_smooth = applySepFilter(hist_norm, flt_G)
        idx = int(mu)
        if idx > 255-3 or idx < 0+3: # 3 because erosion of applySepFilter
            hist_mu = 255
            scale = hist_mu*np.sqrt(2*np.pi)*std
        else:
            alpha = mu - idx
            hist_mu = hist_smooth[idx]*(1 - alpha) + hist_smooth[idx+1]*alpha
            hist_norm_mu = hist_norm_smooth[idx]*(1 - alpha) + hist_norm_smooth[idx+1]*alpha
            scale = hist_mu/hist_norm_mu
    return mu, std, scale

def cdfHisto(hist, Lmax=255):
    '''
    hist: list with length = 256
    cdf: cumulative distribution function
    mapping: convert cdf into 0-255 grayscale levels
    '''
    cumsum = 0
    if isinstance(hist, OrderedDict):
        mapping = OrderedDict()
        cumsum = np.cumsum(list(hist.values() ) )
        hist = list(hist.items())
        coeff = Lmax / cumsum[-1]
        for i, kv in enumerate(hist):
            mapping[kv[0]] = coeff*cumsum[i]
    else:
        cumsum = np.cumsum(hist)
        tolsum = cumsum[-1]
        coeff = Lmax / cumsum[-1]
        mapping = coeff*cumsum
    return mapping

def addOrdereddDict(lhs, rhs):
    lhs = list(lhs.items() )
    rhs = list(rhs.items() )

    dst = []
    l = r = 0
    while True:
        if r == len(rhs):
            if l < len(lhs) :
                dst += lhs[l:]
            break
        elif l == len(lhs):
            if r < len(rhs) :
                dst += rhs[r:]
            break
        else:
            if lhs[l][0] < rhs[r][0]:
                dst.append((lhs[l][0], lhs[l][1]) )
                l += 1
            elif lhs[l][0] == rhs[r][0]:
                dst.append((lhs[l][0], lhs[l][1] + rhs[r][1]) )
                l += 1
                r += 1
            elif lhs[l][0] > rhs[r][0]:
                dst.append((rhs[r][0], rhs[r][1]) )
                r += 1
    return  OrderedDict(dst)

def subOrdereddDict(lhs, rhs):
    dst = []
    l = r = 0
    for rk, rv in rhs.items():
        if rk not in lhs.keys():
            raise KeyError("subOrdereddDict, rhs key {} is not in lhs!\n".format(rk))
        lhs[rk] -= rhs[rk]
    return lhs

def calcHist(src, hist_type='list'):
    '''
    histogram for grayscale image, intensity level is 256

    Parameters
    ----------
    hist_type : string like
        two types of hist_type to define histogram
        - 'list': return hist as length=256 list, `hist[i]` to store the
          times of intensity `i` occurs
        - 'OrderedDict': return hist as length<=256 dict, `hist[k]` to store
          the times of intensity `k` occurs, and will convert to OrderedDict
          and sorted, for the sake of cdfHisto computation
    '''
    htypes = ['list', 'OrderedDict']
    if hist_type not in htypes:
        raise ValueError("Input hist_type {} not in the supported list: {}!\n".format(hist_type, str(htypes)))
    src = np.array(src)
    if src.dtype != np.uint8:
        raise ValueError("Histogram only supports single channel grayscale image, src's dtype is {}".format(repr(src.dtype)))

    sz = src.size
    arr = src.flatten()
    hist = None
    if hist_type == 'list':
        hist = np.zeros(256, dtype=np.int32)
        for i in range(sz):
            hist[ arr[i] ] += 1
    elif hist_type == 'OrderedDict':
        hist = {}
        for i in range(sz):
            if arr[i] not in hist.keys():
                hist[arr[i] ] = 0
            hist[arr[i] ] += 1
        hist = OrderedDict(sorted(hist.items(), key=lambda kv: kv[0]))
    return hist

def calcHistSeries(series_, ranges=RMS_BIN_RANGES, column=None):
    """
    calculate the histogram of given Pandas Serires

    Parameters
    ----------
    series_ : Pandas Series
        object to calculate Histogram fro
    ranges : array like
        Array of the histogram bin boundaries.

    Returns
    -------
    histDF : DataFrame of the hists
    """

    labels = []
    filts = []
    hists = np.zeros((len(series_), len(ranges)))
    nitems = len(ranges)
    log.debug("ranges, {}\n".format(str(ranges)) )

    for i in range(nitems):
        if i == nitems - 1:
            label = "{}~".format(ranges[i])
            filt  = lambda x, y=ranges[i]: (x >= y) & True
        else:
            label = "{}~{}".format(ranges[i], ranges[i+1])
            filt  = lambda x, y=ranges[i], z=ranges[i+1]: (x >= y) & (x < z)
        labels.append(label)
        filts.append(filt)
    log.debug(', '.join(labels))
    log.debug("{}".format(str([ff==filts[0] for ff in filts])) )

    for i, series in enumerate(series_):
        if column is not None:
            try:
                series = series.ix[:, column]
            except KeyError:
                raise KeyError("Input column {} is not in columns {}".format(series.columns))
        if not isinstance(series, pd.Series):
            try:
                series = pd.Series(series)
            except ValueError:
                raise ValueError("Error occurs when converting input into Pandas Series")
        series = series.sort_values()
        log.debug("series_: {}".format(str(series.values )))
        for j, filt in enumerate(filts):
            hists[i, j] = len(series.ix[ series.apply(filt) ] )
            log.debug("{}: {}".format(labels[j], hists[i, j]) )
    histDF = pd.DataFrame(hists, columns=labels).astype('int32')
    histDF = histDF.transpose() # for barplot, column as range lables
    return histDF

def im_fft_amplitude_phase(im, freqshift=True, method=None, raw_amplitude=False):
    '''get image DFT amplitude & phase for the input image by fft'''
    if method is None:
        method = 'fft'
    if method == 'fft':
        fourierfunc = np.fft.fft2
    elif method == 'dft':
        sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../signal")
        from filters import dft
        fourierfunc = dft
    if not freqshift:
    # gen transform matrix T, so fft spectrum is zero-centered
        sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../signal")
        T = np.zeros(im.shape, im.dtype)
        nrows, ncols = im.shape
        for y in range(nrows):
            for x in range(ncols):
                T[y, x] = (-1)**(x+y)
        im = np.multiply(im, T)

    # fft, and rescale the amplitude
    imfft = fourierfunc(im)
    if freqshift:
        imfft = np.fft.fftshift(imfft)
    rawamplitude = np.abs(imfft)
    amplitude  = np.log(1+ rawamplitude) # refer to DIPum
    # amplitude  = 1 + np.log(amplitude) # DIP form
    phase = np.angle(imfft)
    if raw_amplitude:
        amplitude = rawamplitude
    return (amplitude, phase)

def power_ratio_in_cutoff_frequency(amplitude, cutoff):
    '''
    A way to compute the cutoff frequency loci for LPF

    the ratio is defined as:

        r = 100 * Sum( P{inside D0} ) / Sum( P{image power} )

    Parameters
    ----------
    amplitude : 2D array
        image frequency amplitude or
        fourier transform result as complex data type
    cutoff : float or list
        cutoff frequency

    Returns:
    --------
    ratio : float or list
        image power ratio inside cutoff frequency
    '''
    if 'complex' in str(amplitude.dtype):
        power = np.abs(amplitude * np.conjugate(amplitude))
    else:
        power = amplitude**2

    from FrequencyFlt import distance_map
    D0s = [cutoff] if np.ndim(cutoff) == 0 else cutoff
    D = distance_map(amplitude.shape)
    ratios = []
    for D0 in D0s:
        mask = ~(D <= D0)
        power_m = np.ma.array(power, mask=mask)
        ratio = 100 * np.sum(power_m) / np.sum(power)
        ratios.append(ratio)
    ratio = ratios[0] if np.ndim(cutoff) == 0 else ratios
    return ratio

def calculate_cutoff(amplitude, samples=None, thres=95):
    """
    calculate cutoff frequency at specific power ratio thres

    Parameters
    ----------
    samples : 1D array-like
        The cut-off frequencies(radius), elements should <= 1/2 amplitude size
    thres : float or array
        thres ratio value in range of [0, 100]

    Returns
    -------
    ret : float
        the cutoff frequency at specific power ratio
    """
    shape = amplitude.shape
    maxSz = max(shape)/2 # radius
    if samples is None:
        samples = np.linspace(maxSz/10, maxSz, 50)
    ratios = power_ratio_in_cutoff_frequency(amplitude, samples)

    thresholds = [thres] if np.ndim(thres) == 0 else thres
    rets = []
    ridx = 0
    for th in thresholds:
        foundidx = False
        for i in range(len(ratios) - 1):
            if (ratios[i] - th) * (ratios[i+1] - th) <= 0:
                ridx = i
                foundidx = True
                break
        ret = samples[-1]
        if foundidx:
            if ratios[ridx] == ratios[ridx+1]:
                ret = samples[ridx]
            else: # bilinear interpolation
                alpha = (th - ratios[ridx])/(ratios[ridx+1] - ratios[ridx])
                if alpha<0 or alpha>1:
                    raise ValueError("ridx: {}, alpha {}, thresh {}, ratios {} {}\n".format(ridx, alpha, th, ratios[ridx], ratios[ridx+1]))
                ret = (1-alpha)*samples[ridx] + alpha*samples[ridx+1]

        else:
            if th < ratios[0]:
                ret = th/ratios[0]*samples[0]
            elif th > ratios[-1]:
                ret = th/ratios[-1]*samples[-1]
        rets.append(ret)
    rets = np.array(rets)
    rets = np.clip(rets, 0, maxSz)
    ret = rets[0] if np.ndim(thres) == 0 else rets
    return ret

def computeSNR(src, restored):
    """
    compute the SNR of the restored image[s]
        SNR = mean(f^2) / mean((f - fr)^2) # higher SNR, the better, fr -> f
        SNR = mean(fr^2) / mean((g - fr)^2) # less filters the better, not correct, so not use this function now
    """
    im_restore = [restored] if np.ndim(restored) == 0 else restored
    SNRs = []
    for f_r in im_restore:
        imdiff = f_r - src
        SNRs.append(np.mean(f_r**2)/np.mean(imdiff**2))
    ret = SNRs[0] if np.ndim(restored) == 0 else SNRs
    return ret

if __name__ == '__main__':
    print(statHist(cv_gaussian_kernel(3)))
    print(statHist(cv_gaussian_kernel(5)))
    print(statHist(cv_gaussian_kernel(5, 0.7)))
    print(statHist(cv_gaussian_kernel(7)))
