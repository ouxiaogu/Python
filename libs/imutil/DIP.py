"""
-*- coding: utf-8 -*-
Created: peyang, 2018-07-02 11:44:17

Test the Code in DIP:

Last Modified by: ouxiaogu
"""

import cv2
import numpy as np
import sys
from ImGUI import imshowCmap, cvtFloat2Gray
from ImFilters import *
from ImGUI import imshowMultiple, imshowMultiple_TitleMatrix

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
    sys.path.append("../common")
    # from PlatformUtil import home
    # import os
    # imfile = os.path.join(home(), r"github\OpenCV-3.3.1\samples\data\building.jpg")
    imfile = r'C:\Localdata\D\Book\DIP\DIP\imagesets\DIP3E_Original_Images_CH04\Fig0417(a)(barbara).tif'
    im = cv2.imread(imfile, 0)
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
    sys.path.append(r'../signal')
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
    from ImDescriptors import im_fft_amplitude_phase
    prefix = "{}(fftshift = {})".format(method, fftshift)

    # raw image
    im = cv2.imread(r'C:\Localdata\D\Book\DIP\DIP\imagesets\DIP3E_Original_Images_CH04\Fig0424(a)(rectangle).tif', 0)
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

def try_fft_power():
    from ImDescriptors import im_fft_amplitude_phase
    from ImGUI import imshowMultiple
    prefix='FFT'

    # raw image
    im = cv2.imread(r'C:\Localdata\D\Book\DIP\DIP\imagesets\DIP3E_Original_Images_CH04\Fig0441(a)(characters_test_pattern).tif', 0)
    rows, cols = im.shape
    amplitude, phase = im_fft_amplitude_phase(im)

    # plot
    imshowMultiple([im, amplitude, phase], titles=['im','{} amplitude'.format(prefix), '{} phase'.format(prefix)])

def try_filter(option=None):
    from ImFilters import imApplyFilter
    if option is None:
        option = 'LPF'
    if option == 'LPF':
        funcs = [ILPF, BLPF, GLPF]
    elif option == 'HPF':
        funcs = [IHPF, BHPF, GHPF]
    funcname = [f.__name__ for f in funcs]

    # raw image
    im = cv2.imread(r'D:\book\DIP\DIP\imageset\DIP3E_Original_Images_CH04\Fig0442(a)(characters_test_pattern).tif', 0)
    imgs = [im]
    titles = ['raw image']
    for D0 in [10, 30, 60, 160, 460]:
        kwargs = {'D0': D0}
        imgs.append(imApplyFilter(im, ILPF, **kwargs) )
        titles.append('ILPF, D0 = {}'.format(D0))
    imshowMultiple(imgs, titles)


if __name__ == '__main__':
    # try_paramid()

    #import itertools
    # for fftshift, method in itertools.product([True, False],  ['fft', 'dft']):
    # for fftshift in [True, False]:
    #     try_fft(fftshift)

    try_filter()