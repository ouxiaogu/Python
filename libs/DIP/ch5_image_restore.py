# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-07-22 21:30:20

image degradation and restoration

Last Modified by: ouxiaogu
"""

import numpy as np
import cv2

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../imutil")
from ImGUI import *
from ImDescriptors import im_fft_amplitude_phase, hist_rect, printImageInfo, hist_lines, hist_curve
from ImTransform import normalize, intensityTransform, calcHist, imSub, equalizeHisto, powerFunc
from SpatialFlt import ContraHarmonicMean, adpMean, adpMedian, applyMeanFilter, TrimmedMean, setNLMParams
from FrequencyFlt import BNRF, BNPF, applyFreqFilter
KWARGS = {'vmin': 0, 'vmax': 255}

DIPPATH = r'C:\Localdata\D\Book\DIP\DIP\imagesets\DIP3E_Original_Images_CH05'
PROJECTPATH = r'C:\Localdata\D\Book\DIP\DIPum\DIPUM2E_Projects\SAMPLE_DIPUM2E_PROJECT_IMAGES'
# PROJECTPATH = r'D:\book\DIP\DIPum\DIPUM2E_Projects\SAMPLE_DIPUM2E_PROJECT_IMAGES'
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

def try_pepper_salt():
    IMFILE = os.path.join(DIPPATH, r'Fig0508(b)(circuit-board-salt-prob-pt1).tif')
    imsalt = cv2.imread(IMFILE, 0)
    IMFILE = os.path.join(DIPPATH, r'Fig0508(a)(circuit-board-pepper-prob-pt1).tif')
    impepper = cv2.imread(IMFILE, 0)

    imsalt_m = cv2.medianBlur(imsalt, 3)
    imsalt_ch = ContraHarmonicMean(imsalt, 3, -1.5)

    impepper_m = cv2.medianBlur(impepper, 3)
    impepper_ch = ContraHarmonicMean(impepper, 3, 1.5)

    imshowMultiple_TitleMatrix([imsalt, imsalt_m, imsalt_ch] +
        [impepper, impepper_m, impepper_ch], 2, 3,
        ['salt', 'pepper'], ['raw', 'median', 'contra harmonic'],
        cbar=False)

def try_pepper_salt2():
    IMFILE = os.path.join(PROJECTPATH, r'FigP0502(a)(salt_only).tif')
    imsalt = cv2.imread(IMFILE, 0)
    IMFILE = os.path.join(PROJECTPATH, r'FigP0502(b)(pepper_only).tif')
    impepper = cv2.imread(IMFILE, 0)

    imsalt_m = cv2.medianBlur(imsalt, 5)
    imsalt_ch = ContraHarmonicMean(imsalt, 3, -5)
    imsalt_m = cv2.medianBlur(imsalt, 5)
    imsalt_adpm = adpMedian(imsalt, 3, 7)

    impepper_m = cv2.medianBlur(impepper, 5)
    impepper_ch = ContraHarmonicMean(impepper, 3, 1.5)
    impepper_adpm = ContraHarmonicMean(impepper, 3, 7)

    imshowMultiple_TitleMatrix([imsalt, imsalt_m, imsalt_ch, imsalt_adpm] +
        [impepper, impepper_m, impepper_ch, impepper_adpm], 2, 4,
        ['salt', 'pepper'], ['raw', 'median', 'contra harmonic', 'adaptive median'],
        cbar=False)

def try_polyroi_noise_hist(interative=True):
    KEY_ESC = 27

    # IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p3613_LDose.bmp')
    # IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p3613_regular.bmp')
    IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p1521_LDose.bmp')
    #IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p1521_regular.bmp')
    im = cv2.imread(IMFILE, 0)
    # im = cv2.fastNlMeansDenoising(im, h=30,  templateWindowSize=11, searchWindowSize=35)

    if interative:
        window_name = r"draw rect roi"
        pd = RectangleDrawer(im, window_name)
        imroi = pd.run()
        roi = pd.getROI()
    else:
        # vertexes = [(62,189), (31,203), (18,244), (68,283), (87,216)]
        # vertexes = [(140, 599), (878, 599), (878, 662), (140, 662)] # rect
        # imroi = cv2.polylines(im, np.array([vertexes]), True, [0, 255, 255], 1)
        # roi = getPolyROI(im, vertexes)
        pair = [(171, 565), (467, 672)] # rect
        tl, br = pair
        imroi = cv2.rectangle(im, tl, br, (0, 255, 255))
        roi = getROIByPointPairs(im, [pair], cv2.rectangle)
    printImageInfo(roi)
    roi = roi.astype('uint8')
    roihist = calcHist(roi)
    printImageInfo(roihist)
    imhist = hist_rect(hist=roihist, color_hist=True)
    imshowMultiple([imroi, imhist], ['imroi', 'imhist'])

def try_denoise_ldose_various_methods():
    IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p3613_LDose.bmp')
    im = cv2.imread(IMFILE, 0)
    # im = cv2.pyrDown(im)

    im_med = cv2.medianBlur(im, 5)

    im_adpMed = adpMedian(im, 3, 9)

    # im_NonLMean = cv2.fastNlMeansDenoising(im, h=30, templateWindowSize=11, searchWindowSize=35)

    imsalt_ch = ContraHarmonicMean(im, 5, -1.5) # X: not salt for low dose
    impepper_ch = ContraHarmonicMean(im, 5, 1.5) # yes

    im_mean = applyMeanFilter(im, 5)
    im_adpMean = adpMean(im, 5, noise_var=2500)

    im_triMean = TrimmedMean(im, 5, 4)

    # imshowMultiple([im, im_med, im_adpMed, im_NonLMean, impepper_ch, im_mean, im_adpMean, im_triMean],
    #     ['raw', 'median 5x', 'adaptive median 3x, kSzMax=9', 'Non-local mean, h=30,psize=9,bsize=35', 'contra harmonic pepper, 5x, power=1.5',  'mean 5x', 'adpMean 5x, sigmaN=50', 'trimmed mean 5x, d=4'])
    imshowMultiple([im, im_med, im_adpMed, imsalt_ch, impepper_ch, im_mean, im_adpMean, im_triMean],
        ['raw', 'median 5x', 'adaptive median 3x, kSzMax=9', 'contra harmonic pepper, 5x, power=-1.5', 'contra harmonic pepper, 5x, power=1.5',  'mean 5x', 'adpMean 5x, sigmaN=50', 'trimmed mean 5x, d=4'])

def try_denoise_ldose_deepen_one_method(mode='adpMean'):
    # IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p3613_regular.bmp')
    IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p3613_LDose.bmp')
    im = cv2.imread(IMFILE, 0)
    imgs = [im]
    titles = ['raw image']
    ksz = 5

    if mode == 'adpMean':
        iteritems = np.linspace(50, 100, 6)
        iterkey = 'sigma'
    elif mode == 'TrimmedMean':
        iteritems = np.linspace(2, 22, 6)
        iterkey = 'd'
    elif mode.lower() == 'mean':
        iteritems = ['box', 'Gaussian']
        iterkey = 'ktype'

    for item in iteritems:
        kwargs = {'ksz':ksz, iterkey: item}
        argstrs = ['='.join(map(str, kw) ) for kw in zip(kwargs.keys(), kwargs.values()) ]
        titles.append(mode+' '+ ', '.join(argstrs))
        if mode == 'adpMean':
            dst = adpMean(im, ksz, noise_var=item**2)
        elif mode == 'TrimmedMean':
            dst = TrimmedMean(im, ksz, d=item, ktype='box')
        elif mode == 'mean':
            dst = applyMeanFilter(im, ksz, ktype=item)
        imgs.append(dst)

    titles.append('diff last {} and raw image'.format(mode))
    imgs.append(imgs[-1] - imgs[0] )
    imshowMultiple(imgs, titles, **KWARGS)

def try_adpMean():
    IMFILE = os.path.join(DIPPATH, r'Fig0513(a)(ckt_gaussian_var_1000_mean_0).tif')
    im = cv2.imread(IMFILE, 0)

    imMean = applyMeanFilter(im, 7)
    imAdpMean = adpMean(im, 7, noise_var=1000)

    imshowMultiple([im, imMean, imAdpMean],
        ['raw', 'mean', 'adaptive mean'])

def try_adpMedian():
    IMFILE = os.path.join(DIPPATH, r'Fig0514(a)(ckt_saltpep_prob_pt25).tif')
    im = cv2.imread(IMFILE, 0)

    imMed = cv2.medianBlur(im, 7)
    imAdpMed = adpMedian(im, 3, 9)

    imshowMultiple([im, imMed, imAdpMed],
        ['raw', 'median', 'adaptive median'])

def try_NLM():
    # IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p3613_regular.bmp')
    IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p3613_LDose.bmp')
    im = cv2.imread(IMFILE, 0)
    imgs = [im]
    titles = ['raw image']

    # sigma = 0-15-30, patch 3~5, h=0.40*sigma, block = 21
    # sigma = 30-45-75, patch 7~9, h=0.35*sigma, block = 35
    # sigma = 75-100, patch 11, h=0.3*sigma, block = 35
    sigmas = np.linspace(50, 100, 6)
    for sigma in sigmas:
        h, psize, bsize = setNLMParams(sigma)
        kwargs = {'sigma': sigma, 'h': h, 'psize': psize, 'bsize': bsize}
        argstrs = ['='.join(map(str, kw) ) for kw in zip(kwargs.keys(), kwargs.values()) ]
        dst = cv2.fastNlMeansDenoising(im, None, h, psize, bsize)
        imgs.append(dst)
        titles.append('NLM '+ ', '.join(argstrs))
    titles.append('diff last NLM and raw image')
    imgs.append(imgs[-1] - imgs[0] )
    imshowMultiple(imgs, titles)


def try_notch(interative=True):
    KEY_ESC = 27

    IMFILE = os.path.join(PROJECTPATH, r'FigP0405(HeadCT_corrupted).tif')
    im = cv2.imread(IMFILE, 0)
    amp, _ = im_fft_amplitude_phase(im)
    amp = intensityTransform(normalize(amp, 255, np.uint8), powerFunc(gamma=2), dtype=np.uint8)
    if interative:
        window_name = r"draw notch points"
        pd = PolygonDrawer(amp, window_name)
        amproi = pd.run()
        vertexes = pd.points
    else:
        #vertexes = [(216, 217), (246, 256), (256, 276)]
        #vertexes = [(256, 266), (276, 256), (296, 296)] #DIPum
        vertexes = [(296, 296), (256, 276), (265, 257)]
        #[256 266; 276 256; 296 296]
        amproi = cv2.polylines(amp, np.array([vertexes]), True, [0, 255, 255], 1)
    kwargs = {'notches': vertexes,'D0s': 10, 'n':5, 'notch_after_padding':False}

    sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../signal")
    from filters import padding_backward
    rawShape = im.shape
    fp = padding_backward(im, rawShape)
    amplitude, _  = im_fft_amplitude_phase(fp)

    H = BNRF(fp.shape, **kwargs)
    Fp = np.fft.fft2(fp)
    Fp = np.fft.fftshift(Fp)
    Gp = Fp*H
    Gp_A = np.log(1 + np.absolute(Gp))
    imshowMultiple([fp, amplitude, H, Gp_A], ['im', 'fft', 'BNRF', 'frequency result'])

    imBNRF = applyFreqFilter(im, BNRF, **kwargs)
    imBNPF = im - imBNRF
    # imBNPF = imSub(im, imBNRF, Imax=255) # applyFreqFilter(im, BNPF, **kwargs)
    imshowMultiple([im, amproi, imBNRF, imBNPF], ['imroi', 'imhist', 'BNRF', 'BNPF'])

def try_estimate_degrationFunc():
    IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p1521_LDose.bmp')
    g = cv2.imread(IMFILE, 0)

    IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p1521_regular.bmp')
    f = cv2.imread(IMFILE, 0)

def main():
    # try_noise_fft()

    # try_noise()


    # try_pepper_salt()
    # try_pepper_salt2()

    # try_adpMean()
    # try_adpMedian()


    # try_polyroi_noise_hist(True)
    # try_denoise_ldose_various_methods()
    # try_NLM()
    try_denoise_ldose_deepen_one_method('adpMean')

    # try_notch(False)

if __name__ == '__main__':
    main()