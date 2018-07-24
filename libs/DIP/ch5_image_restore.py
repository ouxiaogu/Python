# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-07-22 21:30:20

image degradation and restoration

Last Modified by: ouxiaogu
"""

import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../imutil")
from ImGUI import *
from ImDescriptors import im_fft_amplitude_phase, hist_rect, printImageInfo, hist_lines, hist_curve
from ImTransform import normalize, intensityTransform, calcHist
from SpatialFlt import ContraHarmonic

DIPPATH = r'C:\Localdata\D\Book\DIP\DIP\imagesets\DIP3E_Original_Images_CH05'
PROJECTPATH = r'C:\Localdata\D\Book\DIP\DIPum\DIPUM2E_Projects\SAMPLE_DIPUM2E_PROJECT_IMAGES'
WORKDIR = r"C:\Localdata\D\Note\Python\misc\iCal\SEM\samples"

def try_noise_fft():
    shape = (300, 300)
    gn = np.random.normal(0.5, 0.2, shape )
    printImageInfo(gn)
    amp, _ = im_fft_amplitude_phase(gn)
    printImageInfo(amp)
    araw, _ = im_fft_amplitude_phase(gn, raw_amplitude=True)
    amax = np.amax(araw)
    araw = intensityTransform(araw, lambda x: 0 if x==amax else x)
    power = araw * araw

    sHist = hist_lines( normalize(gn, 255, np.uint8))
    fHist = hist_rect( normalize(power, 255, np.uint8), 100)

    imshowMultiple_TitleMatrix([gn, amp]+ [sHist, fHist], 2, 2,
        ['Gaussian Noise', 'Amplitude'], ['sptial', 'frequency'])

def try_noise():
    histfunc = lambda im: hist_rect(normalize(im, 255, np.uint8), 100)

    shape = (300, 300)
    GaussianNoise = np.random.normal(0.5, 0.2, shape ) # u, sigma
    RayleighNoise = np.random.rayleigh(0.2, shape ) # sigma, a=0 here
    GammaNoise = np.random.gamma(5, 0.2, shape ) # b, sigma = 1/a
    noise1 = [GaussianNoise, RayleighNoise, GammaNoise]
    hist1 = list(map(histfunc, noise1))
    imshowMultiple_TitleMatrix(noise1+hist1, 2, 3,
        ['noise', 'histogram'], ['Gaussian', 'Rayleigh', 'Gamma'],
        cbar=False)

    ExpNoise = np.random.exponential(0.2, shape ) # sigma = 1/a
    UniformNoise = np.random.uniform(0, 255, shape ) # sigma = 1/a
    from scipy.stats import rv_discrete
    pdf = np.zeros(256)
    pdf[0] = 0.1
    pdf[255] = 0.1
    pdf[127] = 0.8
    random_variable = rv_discrete(values=(np.arange(256), pdf))
    ImpulseNoise = random_variable.rvs(size=shape)
    noise2 = [ExpNoise, UniformNoise, ImpulseNoise]
    hist2 = list(map(histfunc, noise2))
    imshowMultiple_TitleMatrix(noise2+hist2, 2, 3,
        ['noise', 'histogram'], ['Exponential', 'Uniform', 'Impulse'],
        cbar=False)

def try_polyroi(interative=True):
    KEY_ESC = 27

    # IMFILE = os.path.join(PROJECTPATH, r'FigP0501(noisy_superconductor_image).tif')
    IMFILE = os.path.join(WORKDIR, r'calaveras_v3_LDose_p3544.bmp')
    im = cv2.imread(IMFILE, 0)

    if interative:
        window_name = r"draw poly roi"
        pd = PolygonDrawer(im, window_name)
        imroi = pd.run()
        print(pd.points)
        roi = getPolyROI(im, pd.points)
    else:
        vertexes = [(62,189), (31,203), (18,244), (68,283), (87,216)]
        imroi = cv2.polylines(im, np.array([vertexes], True, [0, 255, 255], 1))
        roi = getPolyROI(im, vertexes)
    roihist = calcHist(roi)
    printImageInfo(roihist)
    imhist = hist_rect(hist=roihist)
    imshowMultiple([imroi, imhist], ['imroi', 'imhist'])

def try_pepper_salt():
    IMFILE = os.path.join(DIPPATH, r'Fig0508(b)(circuit-board-salt-prob-pt1).tif')
    # IMFILE = os.path.join(PROJECTPATH, r'FigP0502(a)(salt_only).tif')
    imsalt = cv2.imread(IMFILE, 0)
    IMFILE = os.path.join(DIPPATH, r'Fig0508(a)(circuit-board-pepper-prob-pt1).tif')
    # IMFILE = os.path.join(PROJECTPATH, r'FigP0502(b)(pepper_only).tif')
    impepper = cv2.imread(IMFILE, 0)

    imsalt_m = cv2.medianBlur(imsalt, 3)
    imsalt_ch = ContraHarmonic(imsalt, 3, -1.5)

    impepper_m = cv2.medianBlur(impepper, 3)
    impepper_ch = ContraHarmonic(impepper, 3, 1.5)

    imshowMultiple_TitleMatrix([imsalt, imsalt_m, imsalt_ch] +
        [impepper, impepper_m, impepper_ch], 2, 3,
        ['salt', 'pepper'], ['raw', 'median', 'contra harmonic'],
        cbar=False)


def main():
    # try_noise_fft()

    # try_noise()

    try_polyroi()

    # try_pepper_salt()

if __name__ == '__main__':
    main()