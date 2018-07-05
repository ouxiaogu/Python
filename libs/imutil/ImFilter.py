"""
-*- coding: utf-8 -*-
Created: peyang, 2018-07-02 11:44:17

Code about Image Filter, includes:

  - Build filter
  - apply filter
  - example usage for filter

Last Modified by: ouxiaogu
"""

import cv2
import numpy as np
import sys
from scipy import signal
from ImPlotUtil import *

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
    from PlatformUtil import home
    import os
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

if __name__ == '__main__':
    try_paramid()