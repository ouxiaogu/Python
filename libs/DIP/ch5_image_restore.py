# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-07-22 21:30:20

image degradation and restoration

Last Modified by: ouxiaogu
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../imutil")
from ImGUI import *
from ImDescriptors import im_fft_amplitude_phase, hist_rect, printImageInfo, hist_lines, hist_curve, calculate_cutoff, getImageInfo, statHist
from ImTransform import normalize, intensityTransform, calcHist, imSub, equalizeHisto, localHistoEqualize, powerFunc, convertTo
from SpatialFlt import ContraHarmonicMean, adpMean, adpMedian, applyMeanFilter, TrimmedMean, setNLMParams
from FrequencyFlt import BNRF, BNPF, applyFreqFilter, GLPF
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../common")
from FileUtil import splitFileName
from PlotConfig import addLegend, getHexColor

sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../signal")
from filters import padding_backward

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

    IMFILE = os.path.join(WORKDIR, r'vs_calaveras_v3_LDose_p3544.bmp')
    # IMFILE = os.path.join(WORKDIR, r'1521_image.pgm')
    im = cv2.imread(IMFILE, 0)

    if IMFILE[-3:] == 'pgm':
        im = im/((np.iinfo(im.dtype).max+1)/(np.iinfo('>u1').max+1))
        im = im.astype('>u1')

    if interative:
        window_name = r"draw rect roi"
        pd = RectangleDrawer(im, window_name)
        #pd = PolygonDrawer(im, window_name)
        imroi = pd.run()
        roi = pd.getROI()
    else:
        # pair = [(471, 207), (555, 809)] # 1521 bk rect
        # pair = [(143, 140), (436, 548)] # common feature rect
        pair = [(140, 599), (878, 662)] # 3613 bk rect
        tl, br = pair
        roi = getROIByPointPairs(im, [pair], cv2.rectangle)
        tmp = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if np.ndim(im) == 2 else im
        imroi = cv2.rectangle(tmp, tl, br, (0, 255, 255))
    printImageInfo(roi)
    roihist = calcHist(roi)
    from scipy.stats import norm

    # roihist, _ = np.histogram(roi, bins=np.arange(257))
    # print(roihist1[80:120:10])
    # print(roihist[80:120:10])
    # np.testing.assert_equal(roihist1, roihist)
    # mu, std = norm.fit(roi)
    mu, std, _ = statHist(roihist, trimmedNum=2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xends = np.linspace(1, 256, 256)
    bar_width = 1/256
    xstarts = xends - bar_width
    # plt.bar(xstarts, roihist, bar_width, alpha=0.6, color='g', label='bk resist line histo')
    plt.hist(roi, bins=256, density=True, alpha=0.6, color='g', label='bk histogram')

    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normfunc = lambda x: 1/(np.sqrt(2*np.pi)*std) * np.exp( - (x-mu)**2/(2*(std)**2 ))
    ax.plot(x, normfunc(x), 'k', linewidth=2, label='pdf, $\mu={:.3f}, \sigma={:.3f}$'.format(mu, std))
    # p = norm.pdf(x, mu, std)
    # plt.plot(x, p, 'k', linewidth=2, label='pdf, $\mu={:.3f}, \sigma={:.3f}$'.format(mu, std))
    addLegend([ax])

    imhist = hist_rect(hist_in=roihist, color_hist=True, fit_hist=True)
    #allhist = hist_rect(im, hbins=256, color_hist=True)
    imshowMultiple([imroi, imhist], ['imroi', 'imhist'])

def try_denoise_ldose_various_methods(wihist=False):
    IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p3613_LDose.bmp')
    im = cv2.imread(IMFILE, 0)
    # im = cv2.pyrDown(im)

    im_med = cv2.medianBlur(im, 5)

    im_adpMed = adpMedian(im, 3, 9)

    im_NonLMean = cv2.fastNlMeansDenoising(im, h=30, templateWindowSize=11, searchWindowSize=35)

    # imsalt_ch = ContraHarmonicMean(im, 5, -1.5) # X: not salt for low dose
    impepper_ch = ContraHarmonicMean(im, 5, 1.5) # yes

    im_mean = applyMeanFilter(im, 5)
    im_adpMean = adpMean(im, 5, noise_var=44**2)

    im_triMean = TrimmedMean(im, 5, 4)

    imgs=[im, im_med, im_adpMed, im_NonLMean, impepper_ch, im_mean, im_adpMean, im_triMean]
    titles = ['raw', 'median 5x', 'adaptive median 3x, kSzMax=9', 'Non-local mean, h=30,psize=9,bsize=35', 'contra harmonic pepper, 5x, power=1.5',  'mean 5x', 'adpMean 5x, sigmaN=44', 'trimmed mean 5x, d=4']
    # imgs = [im, im_med, im_adpMed, imsalt_ch, impepper_ch, im_mean, im_adpMean, im_triMean]
    # titles = ['raw', 'median 5x', 'adaptive median 3x, kSzMax=9', 'contra harmonic pepper, 5x, power=-1.5', 'contra harmonic pepper, 5x, power=1.5',  'mean 5x', 'adpMean 5x, sigmaN=50', 'trimmed mean 5x, d=4']

    infos = [getImageInfo(im) for im in imgs]
    print('\n'.join(map(', '.join, zip(titles, infos))))
    imshowMultiple(imgs, titles, **KWARGS)

    if wihist:
        histimgs = []
        pair = [(140, 599), (878, 662)] # rect
        tl, br = pair
        for im in imgs:
            roi = getROIByPointPairs(im, [pair], cv2.rectangle)
            roihist = calcHist(roi)
            imhist = hist_rect(hist_in=roihist, color_hist=True)
            histimgs.append(imhist)
        imshowMultiple(histimgs, titles)

def try_denoise_ldose_deepen_one_method(mode='adpMean', save=False):
    # IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p3613_regular.bmp')
    IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p3613_LDose.bmp')
    im = cv2.imread(IMFILE, 0)
    imgs = [im]
    titles = ['raw image']
    ksz = 9

    if save:
        dirname, filename, filextn = splitFileName(IMFILE)
        outfile = os.path.join(dirname, filename+'_'+mode+'.'+filextn)
        if mode == 'adpMean':
            dst = adpMean(im, ksz, noise_var=44**2)
        elif mode == 'TrimmedMean':
            dst = TrimmedMean(im, ksz, d=14)
        elif mode.lower() == 'mean':
            dst = applyMeanFilter(im, ksz)
        elif mode == 'NLM':
            sigma = 100
            h, psize, bsize = setNLMParams(sigma)
            dst = cv2.fastNlMeansDenoising(im, None, h, psize, bsize)
        cv2.imwrite(outfile, dst)
        return 0

    if mode == 'adpMean':
        iteritems = np.linspace(50, 100, 6)
        iterkey = 'sigma'
    elif mode == 'TrimmedMean':
        iteritems = np.linspace(2, 22, 6)
        iterkey = 'd'
    elif mode.lower() == 'mean':
        # iteritems = ['box', 'Gaussian']
        # iterkey = 'ktype'
        iteritems = np.linspace(3, 13, 6)
        iterkey = 'ksize'
    elif mode == 'NLM':
        iteritems = np.linspace(60, 160, 6)
        iterkey = 'sigma'
    else:
        raise NotImplementedError("mode {} is not implemented!\n".format(mode))

    for item in iteritems:
        if mode == 'adpMean':
            dst = adpMean(im, ksz, noise_var=item**2)
        elif mode == 'TrimmedMean':
            dst = TrimmedMean(im, ksz, d=item, ktype='Gaussian')
        elif mode == 'mean':
            dst = applyMeanFilter(im, ksz, ktype=item) if iterkey == 'ktype' else applyMeanFilter(im, item, 'box')
        elif mode == 'NLM':
            h, psize, bsize = setNLMParams(item)
            dst = cv2.fastNlMeansDenoising(im, None, h, psize, bsize)
        if mode != 'NLM':
            kwargs = {'ksz':ksz, iterkey: item} if iterkey != 'ksize' else {iterkey: item}
        else:
            kwargs = {iterkey: item, 'h':h, 'psize':psize, 'bsize':bsize}
        argstrs = ['='.join(map(str, kw) ) for kw in zip(kwargs.keys(), kwargs.values()) ]
        titles.append(mode+' '+ ', '.join(argstrs))
        imgs.append(dst)

    titles.append('diff last {} and raw image'.format(mode))
    imgs.append(imgs[-1] - imgs[0] )
    imshowMultiple(imgs, titles, **KWARGS)

def try_denoise_combined():
    IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p3613_LDose.bmp')
    im = cv2.imread(IMFILE, 0)
    im_adpMed = adpMedian(im, 3, 7)
    im_adpMedMean = adpMean(im_adpMed, 5, noise_var=44**2)

    # from scipy import signal
    # im_wiener = signal.wiener(im, 5, noise=44**2)
    # dirname, filename, filextn = splitFileName(IMFILE)
    # outfile = os.path.join(dirname, filename+'_weiner.'+filextn)
    # cv2.imwrite(outfile, im_adpMean)

    imgs = [im, im_adpMed, im_adpMedMean] # im_wiener
    titles = ['raw', 'adaptive median 3x, kSzMax=7', 'adpMean on adpMedian 5x, sigmaN=44'] # , 'wiener 5x, sigmaN=44'

    im_Med = cv2.medianBlur(im, 3)
    im_MedMean = applyMeanFilter(im_Med, 5)
    imgs += [im, im_Med, im_MedMean]
    titles += ['raw', 'median 3x', 'mean on median 5x'] # , 'wiener 5x, sigmaN=44'

    infos = [getImageInfo(im) for im in imgs]
    print('\n'.join(map(', '.join, zip(titles, infos))))
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
    IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p1521_LDose_shappen.bmp')
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
    IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p1521_LDose_P7.bmp')
    g = cv2.imread(IMFILE, 0)

    IMFILE = os.path.join(WORKDIR, r'1521_image.pgm')
    f = read_pgm(IMFILE)
    f = f/((np.iinfo(f.dtype).max+1)/(np.iinfo('>u1').max+1))
    f = f.astype('>u1')
    printImageInfo(f)

    roi = [(143, 140), (436, 548)]
    rc = [(y, x) for x, y in roi]
    myslice = tuple(slice(start, end) for start, end in zip(*rc))

    gs = g[myslice]
    fs = f[myslice]
    padShape = tuple(g.shape[i] - gs.shape[i] for i in range(len(g.shape)) )
    print(gs.shape, g.shape, padShape)
    gs = padding_backward(gs, padShape)
    fs = padding_backward(fs, padShape)
    print(gs.shape, fs.shape)
    GsA, _ = im_fft_amplitude_phase(gs)
    FsA, _ = im_fft_amplitude_phase(fs)

    Gs = np.fft.fft2(gs)
    Fs = np.fft.fft2(fs)
    # import scipy
    # G = scipy.misc.imresize(Gs, g.shape)
    # F = scipy.misc.imresize(Fs, g.shape)
    H = Gs/Fs
    G1 = np.fft.fft2(g)
    F1 = G1/H
    f1 = np.real(np.fft.ifft2(F1))

    IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p1521_LDose.bmp')
    g2 = cv2.imread(IMFILE, 0)
    G2 = np.fft.fft2(g2)
    F2 = G2/H
    f2 = np.real(np.fft.ifft2(F2))

    IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p3613_LDose.bmp')
    g3 = cv2.imread(IMFILE, 0)
    G3 = np.fft.fft2(g3)
    F3 = G3/H
    f3 = np.real(np.fft.ifft2(F3))

    # imshowMultiple([g, f], ['degraded low dose: g', 'averaged: f'])
    # imshowMultiple([gs, fs, GsA, FsA], ['gs', 'fs', 'Gs', 'Fs'])
    imshowMultiple_TitleMatrix([g, g2, g3, f1, f2, f3], 2, 3, ['degraded', 'restored'], ['p1521 p7 complete', 'p1521 p7', 'p3613'])

def try_LPF(save=False):
    # imfiles = [r'Calaveras_v3_p1521_LDose_P7.bmp'] , r'vs_calaveras_v3_LDose_p3544.bmp', r'Calaveras_v3_p3613_LDose.bmp']
    imfiles = [ r'Calaveras_v3_p3613_LDose.bmp']
    for imfile in imfiles:
        IMFILE = os.path.join(WORKDIR, imfile)
        g = cv2.imread(IMFILE, 0)
        im = g
        rawShape = g.shape
        gp = padding_backward(g, rawShape)
        Gp = np.fft.fft2(gp)
        Gp = np.fft.fftshift(Gp)
        Gp_a = np.log(1+np.abs(Gp))

        funcs = [GLPF]
        # thres = np.linspace(80, 95, 6)
        # cutoffs = calculate_cutoff(Gp, thres=thres)
        cutoffs = [96, 100, 268, 585, 780, 935]
        thres = power_ratio_in_cutoff_frequency(Gp, cutoffs)

        thres_cutoffs = list(zip(thres, cutoffs))
        print(imfile)
        print('\n'.join(['{:.2f}%: {:.2f}'.format(th, d) for th, d in thres_cutoffs]))

        # raw image
        imgs = [im]
        titles = ['raw image']
        flt_imgs = []
        flt_titles = []
        n = 2

        import re
        for func in funcs:
            for th, D0 in thres_cutoffs:
                kwargs = {'D0': round(D0, 3)}
                if re.match('^B\w{1}PF', func.__name__):
                    kwargs['n'] = n
                if re.match('^\w{1}B\w{1}F', func.__name__):
                    W = D0*0.4
                    kwargs['W'] = W
                fltIm = applyFreqFilter(im, func, **kwargs)

                kwargs['percentage'] = round(th, 3)
                argstrs = ['='.join(map(str, kw) ) for kw in zip(kwargs.keys(), kwargs.values()) ]
                labels = [func.__name__] + argstrs
                label = ', '.join(labels)

                if save:
                    if th >= 85 and th <= 87:
                        dirname, filename, filextn = splitFileName(IMFILE)
                        outfile = os.path.join(dirname, filename+'_thres_'+str(th)+'.'+filextn)
                        cv2.imwrite(outfile, fltIm)
                flt_imgs.append(fltIm )
                flt_titles.append(label)
            flt_titles.append('diff last {} and raw image'.format(func.__name__))
            flt_imgs.append(flt_imgs[-1] - imgs[0] )

            imshowMultiple(imgs + flt_imgs, titles + flt_titles,**KWARGS)
            flt_imgs.clear()
            flt_titles.clear()

def try_LPF_fit_bk_noise():
    imfiles = [r'Calaveras_v3_p1521_LDose_P7.bmp', r'vs_calaveras_v3_LDose_p3544.bmp', r'Calaveras_v3_p3613_LDose.bmp']
    sigmas = np.array([43.86, 44.0, 43.91])

    rawimgs, flt_imgs = [], []
    rawtitles, flt_titles = [], []
    for i, imfile in enumerate(imfiles):
        IMFILE = os.path.join(WORKDIR, imfile)
        im = cv2.imread(IMFILE, 0)
        rawimgs.append(im)
        rawtitles.append(imfile)

        D0 = np.max(im.shape)/sigmas[i]*2
        # D0 = 100
        kwargs = {'D0': round(D0, 3)}
        fltIm = applyFreqFilter(im, GLPF, **kwargs)
        flt_imgs.append(fltIm)
        flt_titles.append(imfile+', LPF D0={:.3f}'.format(D0))
    diffs = [rawimgs[i] - flt_imgs[i] for i in range(3)]
    difftitles = ['diff '+str(i) for i in range(3)]
    imshowMultiple(rawimgs+flt_imgs+diffs, rawtitles+flt_titles+difftitles, **KWARGS)

def try_LPF_dramatic_cutoff():
    imfiles = [r'Calaveras_v3_p1521_LDose_P7.bmp', r'vs_calaveras_v3_LDose_p3544.bmp', r'Calaveras_v3_p3613_LDose.bmp']
    cutoffs = np.linspace(1, 512, 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, imfile in enumerate(imfiles):
        IMFILE = os.path.join(WORKDIR, imfile)
        g = cv2.imread(IMFILE, 0)
        im = g
        rawShape = g.shape
        # gp = padding_backward(g, rawShape)
        Gp = np.fft.fft2(g)
        Gp = np.fft.fftshift(Gp)

        funcs = [GLPF]
        thres = power_ratio_in_cutoff_frequency(Gp, cutoffs)

        thres_cutoffs = list(zip(thres, cutoffs))
        ax.plot(cutoffs, thres, color=getHexColor(ix=i), label='{}'.format(imfile))
    # ax.set_xlim([0, 512])
    # ax.set_ylim([0, 100])
    ax.set_ylabel("power spectrum percentage")
    ax.set_xlabel("cutoff frequency")
    addLegend([ax])


def try_histeq_localhisteq():
    imfiles = [r'Calaveras_v3_p3613_LDose.bmp', r'Calaveras_v3_p3613_LDose_mean.bmp']
    imgs = []
    titles = []
    for i, imfile in enumerate(imfiles):
        IMFILE = os.path.join(WORKDIR, imfile)
        im = cv2.imread(IMFILE, 0)
        imgs.append(im)
        titles.append(imfile)

        im_histEq = equalizeHisto(im)
        im_histLocalEq = localHistoEqualize(im, ksize=3)
        imgs += [im_histEq, im_histLocalEq]

    imshowMultiple_TitleMatrix(imgs, 2, 3,
        titles, ['raw image', 'histeq', 'localhisteq'])


def main():
    # try_noise_fft()

    # try_noise()


    # try_pepper_salt()
    # try_pepper_salt2()

    # try_adpMean()
    # try_adpMedian()

    # try_polyroi_noise_hist(interative=True)
    # try_denoise_ldose_various_methods(wihist=True)
    # try_denoise_ldose_deepen_one_method(mode='NLM', save=False)

    # try_denoise_combined()
    # try_NLM()

    # try_notch(False)
    # try_LPF(save=False)
    # try_LPF_fit_bk_noise()
    try_LPF_dramatic_cutoff()
    # try_histeq_localhisteq()

if __name__ == '__main__':
    main()