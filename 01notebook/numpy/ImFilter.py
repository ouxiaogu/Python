"""
-*- coding: utf-8 -*-
Created: peyang, 2018-07-02 11:44:17

Code about Image Filter, includes:

  - Build filter
  - apply filter
  - example usage for filter

Last Modified by: peyang
"""

import cv2
import numpy as np
import sys
from scipy import signal

def try_paramid():
    """
    pyramid down:
      1. image DFT
      2. image frequency domain, Gaussian LP
      3. reduce image into half size
    """
    im = cv2.imread(r"..\..\..\github\OpenCV-3.3.1\samples\data\building.jpg", 0)
    if(im is None):
       sys.exit("Failed to read image!\n")
    reduced = cv2.pyrDown(im)
    cv2.imshow("build", im)
    cv2.imshow("reduced", reduced)

    sptr = np.fft.fft2(im)
    sptrShifted = np.fft.fftshift(sptr)
    magShifted = np.absolute(sptrShifted)
    magShifted = np.log(magShifted)

    from ImPlotUtil import imshowCmap
    imshowCmap(magShifted, title="building's frequency magnitude, log2 level")

    ''' # frequency domain convolve flt_G is wrong
    sys.path.append(r'../../algo')
    from filters import convolve, gaussian_filter
    flt_G = gaussian_filter(1)
    sptrLP = convolve(sptr, flt_G)
    sptrLP_Shifted = np.fft.fftshift(sptrLP)
    magLP_Shifted = np.absolute(sptrLP_Shifted)
    magLP_Shifted = np.log(magLP_Shifted)
    imshowCmap(magLP_Shifted, title="building's frequency(LP) magnitude, log2 level")
    sptr_pyrDown = sptr[0:-1:2, 0:-1:2]

    '''

    # frequency domain apply a frequency cutting window
    sptr_w, sptr_h = sptrShifted.shape

    curshape = np.asarray(sptrShifted.shape)
    hlfShape = curshape // 2
    startind = (curshape - hlfShape) // 2
    endind = startind + hlfShape
    shapeslice = tuple(slice(startind[k], endind[k]) for k in range(len(endind)))
    sptr_pyrDown = sptrShifted[shapeslice]

    im_pyrDown = np.fft.ifft2(sptr_pyrDown)
    im_pyrDown = np.absolute(im_pyrDown)
    imshowCmap(im_pyrDown, "freq. domain, step-by-step pyrDown")

    # spatial domain, convolve Gaussian filter, and down-sampling
    # opencv use this one
    sys.path.append(r'../../algo')
    from filters import convolve, gaussian_filter
    flt_G = gaussian_filter(1)
    im_conv = convolve(im, flt_G)
    im_pyrDown = im_conv[0:-1:2, 0:-1:2]
    imshowCmap(im_pyrDown, "spatial domain, step-by-step pyrDown")

    # spatial domain, directly down-sampling
    im_pyrDown = im[0:-1:2, 0:-1:2]
    imshowCmap(im_pyrDown, "spatial domain, stair case down-sampling")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try_paramid()