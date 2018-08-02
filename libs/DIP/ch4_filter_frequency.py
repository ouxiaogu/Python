"""
-*- coding: utf-8 -*-
Created: peyang, 2018-07-02 11:44:17

Test the Code in DIP chapter 4: Filtering in Frequency Domain

Last Modified by: ouxiaogu
"""

import cv2
import numpy as np
import re

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../imutil")
from ImGUI import imshowCmap, cvtFloat2Gray, imshowMultiple, imshowMultiple_TitleMatrix
from FrequencyFlt import *
from ImDescriptors import im_fft_amplitude_phase

DIPPATH = r'C:\Localdata\D\Book\DIP\DIP\imagesets\DIP3E_Original_Images_CH04'
# DIPPATH = r'D:\book\DIP\DIP\imageset\DIP3E_Original_Images_CH04'
WORKDIR = r"C:\Localdata\D\Note\Python\misc\iCal\SEM\samples"

def try_paramid():
    """
    pyramid down:
      1. image DFT
      2. image frequency domain, LP
      3. reduce image into half size

    Test results draw the below conclusions:
      1. cv pyrDown is close to "method 3, spatial domain, convolve Gaussian
      filter & downsample",
      2. best is "method 2, LP in frequency domain", worst is "method 4, directly downsample",
    """

    """method 1, cv2.pyrDown"""
    sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../common")
    # from PlatformUtil import home
    # import os
    # IMFILE = os.path.join(home(), r"github\OpenCV-3.3.1\samples\data\building.jpg")
    im = cv2.imread(IMFILE, 0)
    nrows, ncols = im.shape
    if(im is None):
       sys.exit("Failed to read image!\n")
    reduced = cv2.pyrDown(im)
    cv2.imshow("input", im)
    im_pyrDown = cv2.resize(reduced, (ncols, nrows) )
    cv2.imshow("reduced", im_pyrDown)

    """method 2, LP in frequency domain, apply a frequency cutting window"""
    sptr = np.fft.fft2(im)
    sptrShifted = np.fft.fftshift(sptr)
    magShifted = np.absolute(sptrShifted)
    magShifted = np.log(magShifted)
    imshowCmap(magShifted, title="input im's frequency magnitude, log2 level")

    # sptr_pyrDown = np.zeros(sptrShifted.shape, dtype=complex)
    curshape = np.asarray(sptrShifted.shape)
    hlfShape = curshape // 2
    startind = (curshape - hlfShape) // 2
    endind = startind + hlfShape
    shapeslice = tuple(slice(startind[k], endind[k]) for k in range(len(endind)))
    sptr_pyrDown = sptrShifted[shapeslice]
    im_pyrDown = np.fft.ifft2(sptr_pyrDown)
    im_pyrDown = np.absolute(im_pyrDown)
    im_pyrDown = cv2.resize(im_pyrDown, (ncols, nrows) )
    im_pyrDown = cvtFloat2Gray(im_pyrDown)
    cv2.imshow("freq. domain, step-by-step ideal half window filter", im_pyrDown)
    # imshowCmap(im_pyrDown, "freq. domain, step-by-step ideal half window filter",)

    """method 3, spatial domain, convolve Gaussian filter & downsample"""
    sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../signal")
    from filters import convolve, gaussian_filter
    flt_G = gaussian_filter(1)
    im_conv = convolve(im, flt_G)
    im_pyrDown = im_conv[0:-1:2, 0:-1:2]
    im_pyrDown = cv2.resize(im_pyrDown, (ncols, nrows) )
    im_pyrDown = cvtFloat2Gray(im_pyrDown)
    cv2.imshow("spatial domain, convolve G, downsample 1/2, then resize 2X", im_pyrDown)

    """method 4, spatial domain, directly down-sampling"""
    im_pyrDown = im[0:-1:2, 0:-1:2]
    im_pyrDown = cv2.resize(im_pyrDown, (ncols, nrows) )
    im_pyrDown = cvtFloat2Gray(im_pyrDown)
    cv2.imshow("spatial domain, directly downsample 1/2, then resize 2X", im_pyrDown)

    """method 5, frequency domain, strictly follow DIP frequency domain filter apply steps"""
    # 1. gen transform matrix, so fft spectrum is zero-centered
    transMat = np.zeros(im.shape )
    for y in range(nrows):
        for x in range(ncols):
            transMat[y, x] = (-1)**(x+y)
    imTrans = np.multiply(im, transMat)
    sptrTrans = np.fft.fft2(imTrans)
    sptr_h, sptr_w = sptrTrans.shape
    mag = np.absolute(sptrTrans)
    mag = np.log(mag)
    # imshowCmap(mag, title="step-by-step pyrDown, 1. FFT translated")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def try_fft(fftshift=True, method=None):
    prefix = "{}(fftshift = {})".format(method, fftshift)

    # raw image
    IMFILE = os.path.join(DIPPATH, r'Fig0424(a)(rectangle).tif')
    im = cv2.imread(IMFILE, 0)
    rows, cols = im.shape
    amplitude, phase = im_fft_amplitude_phase(im, fftshift, method)

    # Translate
    M = np.float32([[1,0, 320],[0, 1, -400]] )
    imT = cv2.warpAffine(im, M, (cols, rows) )
    amplitudeT, phaseT = im_fft_amplitude_phase(imT, fftshift, method)

    # rotation
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -45, 1)
    imR = cv2.warpAffine(im, M, (cols,rows) ) # transpose np.shape is cv::Size
    amplitudeR, phaseR = im_fft_amplitude_phase(imR, fftshift, method)

    # plot
    imshowMultiple_TitleMatrix([im, amplitude, phase,
                    imT, amplitudeT, phaseT,
                    imR, amplitudeR, phaseR], 3, 3, ['im', 'im translate', 'im Rotation'], ['spatial', '{} amplitude'.format(prefix), '{} phase'.format(prefix)])

def try_power_ratio_loci():
    from ImGUI import imshowMultiple
    prefix='FFT'

    # raw image
    # im = cv2.imread(r'C:\Localdata\D\Book\DIP\DIP\imagesets\DIP3E_Original_Images_CH04\Fig0441(a)(characters_test_pattern).tif', 0)
    IMFILE = os.path.join(DIPPATH, r'Fig0441(a)(characters_test_pattern).tif')
    im = cv2.imread(IMFILE, 0)

    # padding into 2X
    sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../signal")
    from filters import padding_backward
    rawShape = im.shape
    fp = padding_backward(im, rawShape)
    amplitude, phase = im_fft_amplitude_phase(fp, raw_amplitude=True)

    # plot
    # imshowMultiple([fp, amplitude, phase], titles=['im padding','{} amplitude'.format(prefix), '{} phase'.format(prefix)])

    from ImDescriptors import power_ratio_in_cutoff_frequency
    ratios = [(D0, power_ratio_in_cutoff_frequency(amplitude, D0)) for D0 in np.linspace(10, 600, 8)]
    print(ratios)

def try_filter(option=None):
    from FrequencyFlt import applyFreqFilter
    if option is None:
        option = 'LPF'
    if option == 'LPF':
        funcs = [GLPF]
        # cutoffs = [10, 30, 60, 160, 460]
        # cutoffs = [160, 300, 460]
        cutoffs = np.linspace(30, 60, 7)
    elif option == 'HPF':
        funcs = [ GHPF] # IHPF, BHPF,
        cutoffs = [30, 60, 160, 460, 760]
    elif option == 'BPF':
        funcs = [ GBPF]
        cutoffs = np.linspace(30, 60, 7)

    # raw image
    IMFILE = os.path.join(DIPPATH, r'Fig0442(a)(characters_test_pattern).tif')
    im = cv2.imread(IMFILE, 0)
    im = cv2.medianBlur(im, 3)
    imgs = [im]
    # titles = ['raw image']
    titles = ['image w/i medianBlur']
    flt_imgs = []
    flt_titles = []
    n = 2

    for func in funcs:
        for D0 in cutoffs:
            kwargs = {'D0': D0}
            if re.match('^B\w{1}PF', func.__name__):
                kwargs['n'] = n
            if re.match('^\w{1}B\w{1}F', func.__name__):
                W = D0*0.4
                kwargs['W'] = W

            argstrs = ['='.join(map(str, kw) ) for kw in zip(kwargs.keys(), kwargs.values()) ]
            labels = [func.__name__] + argstrs
            label = ', '.join(labels)
            flt_imgs.append(applyFreqFilter(im, func, **kwargs) )
            flt_titles.append(label)
        imshowMultiple(imgs + flt_imgs, titles + flt_titles)
        flt_imgs.clear()
        flt_titles.clear()

def try_HEF():
    # read image
    IMFILE = os.path.join(DIPPATH, r'Fig0343(a)(skeleton_orig).tif')
    im = cv2.imread(IMFILE, 0)
    # im = cv2.medianBlur(im, 3)
    imgs = [im]
    titles = ['raw image']
    # titles = ['image w/i medianBlur']
    amplitude, _  = im_fft_amplitude_phase(im)
    imgs += [amplitude]
    titles += ['fft']

    cutoffs = [10, 20, 50, 100]
    func = HEF
    subfunc = GHPF
    flt_imgs = []
    flt_titles = []
    for D0 in cutoffs:
        kwargs = {'HPFfunc': subfunc, 'D0': D0, 'k1':1, 'k2':3}
        argstrs = ['='.join(map(str, kw) ) for kw in zip(kwargs.keys(), kwargs.values()) ]
        labels = [func.__name__, subfunc.__name__] + argstrs[1:]
        label = ', '.join(labels)
        flt_imgs.append(applyFreqFilter(im, func, **kwargs) )
        flt_titles.append(label)
    imshowMultiple(imgs + flt_imgs, titles + flt_titles)

def try_homomorphic_filter():
    # raw image
    # im = cv2.imread(r'C:\Localdata\D\Book\DIP\DIP\imagesets\DIP3E_Original_Images_CH04\Fig0462(a)(PET_image).tif', 0)
    im = cv2.imread(IMFILE, 0)

    from FrequencyFlt import imApplyHomomorphicFilter
    kwargs = {'gamma_L': 0.5, 'gamma_H': 1.5, 'c': 1}
    flt_im = imApplyHomomorphicFilter(im, 80, **kwargs)
    imshowMultiple([im, flt_im], ['orig image', 'homomorphic filter'])

def try_display_filter():
    funcs = [ILPF, IHPF, IBRF, IBPF,
            BLPF, BHPF, BBRF, BBPF,
            GLPF, GHPF, GBRF, GBPF]
    shape = (400, 300)
    D0 = 100
    n = 2
    W = 10
    flts = []
    titles = []
    for func in funcs:
        kwargs = {'D0': D0}
        if re.match('^B\w{2}F', func.__name__):
            kwargs['n'] = n
        if re.match('^\w{1}B\w{1}F', func.__name__):
            kwargs['W'] = W
        flts.append(func(shape, **kwargs) )

        argstrs = ['='.join(map(str, kw) ) for kw in zip(kwargs.keys(), kwargs.values()) ]
        labels = [func.__name__] + argstrs
        label = ', '.join(labels)
        titles.append(label)
    imshowMultiple(flts, titles)

def try_notch_filter():
    # raw image
    IMFILE = os.path.join(DIPPATH, r'Fig0464(a)(car_75DPI_Moire).tif')
    im = cv2.imread(IMFILE, 0)

    # padding into 2X
    sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../signal")
    from filters import padding_backward
    rawShape = im.shape
    fp = padding_backward(im, rawShape)
    amplitude, _  = im_fft_amplitude_phase(fp)

    notches = [(108.3, 169.6), (222.1, 161.4), (107.6, 86.6), (222.1, 79.1)] #(x,y)
    D0s = [10 for n in notches]
    kwargs = {'notches': notches, 'D0s': D0s, 'n': 4, 'notch_after_padding':True}
    H = BNRF(fp.shape, **kwargs)
    Fp = np.fft.fft2(fp)
    Fp = np.fft.fftshift(Fp)
    Gp = Fp*H
    Gp_A = np.log(1 + np.absolute(Gp))

    res = applyFreqFilter(im, BNRF, **kwargs)

    imshowMultiple([fp, amplitude, H, Gp_A], ['im', 'fft', 'BNRF', 'frequency result'])
    imshowMultiple([im, res], ['im', 'BNRF result'])


if __name__ == '__main__':
    # try_paramid()

    #import itertools
    # for fftshift, method in itertools.product([True, False],  ['fft', 'dft']):

    # for fftshift in [True]: #False
    #    try_fft(fftshift)

    try_power_ratio_loci()

    # try_filter('LPF')
    # try_filter('HPF')
    # try_filter('BPF')

    # try_HEF()

    # try_homomorphic_filter()

    # try_display_filter()

    # try_notch_filter()