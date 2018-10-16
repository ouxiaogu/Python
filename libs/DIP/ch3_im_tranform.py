# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-07-16 11:26:04

DIP Image intensity transform and spatial filter practice

Last Modified by:  ouxiaogu
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../imutil")
from ImTransform import *
from SpatialFlt import *
from ImGUI import imshowMultiple_TitleMatrix, imshowMultiple
from ImDescriptors import hist_lines
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../signal")
from filters import fftconvolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
from FileUtil import splitFileName

import cv2
import matplotlib.pyplot as plt

# INPATH = r'C:\Localdata\D\Note\Python\misc\iCal\SEM\samples'
# DIPPATH = r'C:\Localdata\D\Book\DIP\DIP\imagesets\DIP3E_Original_Images_CH03'
DIPPATH = r'D:\book\DIP\DIP\imageset\DIP3E_Original_Images_CH03'
WORKDIR = r"C:\Localdata\D\Note\Python\misc\iCal\SEM\samples"
# WORKDIR = r"D:\code\Python\apps\MXP\samples"
KWARGS = {'vmin': 0, 'vmax': 255}


def try_equalizeHisto():
    imnames = ['Fig0316(4)(bottom_left).tif', 'Fig0316(1)(top_left).tif', 'Fig0316(2)(2nd_from_top).tif', 'Fig0316(3)(third_from_top).tif', ]

    imfiles = [os.path.join(DIPPATH, name) for name in imnames]
    col_titles = ['dark', 'bright', 'low contrast', 'large gray range']

    images = [cv2.imread(f, 0) for f in imfiles]
    hist_rect0 = [hist_lines(im) for im in images]
    outputs = [equalizeHisto(im) for im in images]
    hist_rect1 = [hist_lines(im) for im in outputs]

    imshowMultiple_TitleMatrix(images+hist_rect0+outputs+hist_rect1,
        4, 4, ['raw', 'raw histo', 'equalizeHisto',  'equalized histo'],
        col_titles, cbar=False, **KWARGS)

def try_equalizeHisto_LowDose():
    infile = os.path.join(WORKDIR, r'calaveras_v3_LDose_p3544.bmp')

    im = cv2.imread(infile, 0)
    im = cv2.medianBlur(im, 5)
    hist = hist_lines(im)

    equalizedIm = equalizeHisto(im)
    equalizedHisto = hist_lines(equalizedIm)

    imshowMultiple_TitleMatrix([im, hist, equalizedIm, equalizedHisto],
        2, 2, ['raw', 'equalized'], ['image', 'Histogram']
        , cbar=False, **KWARGS)

def try_specifyHisto():
    sfile = os.path.join(WORKDIR, r'Calaveras_v3_p3613_LDose.bmp')
    rfile = os.path.join(WORKDIR, r'Calaveras_v3_p3613_regular.bmp')

    ksize = 5
    src = cv2.imread(sfile, 0)
    src = cv2.medianBlur(src, ksize)
    shist = hist_lines(src)

    ref = cv2.imread(rfile, 0)
    ref = cv2.medianBlur(ref, ksize)
    rhist = hist_lines(ref)
    refE = equalizeHisto(ref)
    rhistE = hist_lines(refE)

    dst = specifyHisto(src, ref)
    dhist = hist_lines(dst)
    dstE = specifyHisto(src, refE)
    dhistE = hist_lines(dstE)

    imshowMultiple_TitleMatrix([src, ref, dst, refE, dstE] + [shist, rhist, dhist, rhistE, dhistE],
        2, 5, ['image', 'Histogram'], ['Low dose', 'regular', 'specified', 'reg equalized', 'specified to regular equalized'],
        cbar=False, **KWARGS)

def try_localHistoEq():
    # Elapsed time for equalizeHisto: 0.38696893308588187!
    # Elapsed time for localHistoEqualize1: 15.967883523724595!
    # Elapsed time for localHistoEqualize2: 14.640780619891302!
    # Elapsed time for localHistoEqualize3: 24.334276688725367!
    infile = os.path.join(DIPPATH, r'Fig0326(a)(embedded_square_noisy_512).tif')
    im = cv2.imread(infile, 0)
    hist = hist_lines(im)

    import timeit

    start = timeit.default_timer()
    imE = equalizeHisto(im)
    histE = hist_lines(normalize(imE, 255, np.uint8))
    end = timeit.default_timer()
    print("Elapsed time for equalizeHisto: {}!".format(end - start) )

    start = timeit.default_timer()
    imLE1 = localHistoEqualize1(im)
    histLE1 = hist_lines(imLE1)
    end = timeit.default_timer()
    print("Elapsed time for localHistoEqualize1: {}!".format(end - start) )

    start = timeit.default_timer()
    imLE2 = localHistoEqualize2(im)
    histLE2 = hist_lines(imLE2)
    end = timeit.default_timer()
    print("Elapsed time for localHistoEqualize2: {}!".format(end - start) )

    start = timeit.default_timer()
    imLE3 = localHistoEqualize3(im)
    histLE3 = hist_lines(imLE3)
    end = timeit.default_timer()
    print("Elapsed time for localHistoEqualize3: {}!".format(end - start) )

    imshowMultiple_TitleMatrix([im, imE, imLE2, imLE3]+
        [hist, histE, histLE2, histLE3], 2, 4,
        ['image', 'Histogram'],
        ['raw', 'equalized', 'local equalized 2', 'local equalized 3'],
        cbar=False, **KWARGS)

def try_Laplace_LoG():
    infile = os.path.join(DIPPATH, r'Fig0338(a)(blurry_moon).tif')
    im = cv2.imread(infile, 0)

    fltShape = (3, 3)

    flt_L = LaplaceFilter(fltShape)
    imL = fftconvolve(im, flt_L)
    imL_norm = normalize(imL)

    flt_LoG = LoG(fltShape)
    imLoG = fftconvolve(im, flt_LoG)
    imLoG_norm = normalize(imLoG)

    imLD_norm = normalize(imLoG - imL)

    imshowMultiple([im, imL, imL_norm, imLoG, imLoG_norm, imLD_norm],
        ['raw', 'Laplace', 'Laplace Norm', 'LoG', 'LoG Norm', 'imLoG - imL'], **KWARGS)

    fltShape = (5, 5)
    flt_L2 = LaplaceFilter(fltShape)
    imL2 = fftconvolve(im, flt_L2)
    imL2_norm = normalize(imL2)

    imLD_norm = imL2 - imL

    imshowMultiple([im, imL, imL_norm, imL2, imL2_norm, imLD_norm],
        ['raw', 'Laplace 3x3', 'L 3 Norm', 'Laplace 5x5', 'L 5 Norm', 'im L5 - im L3'], **KWARGS)

def try_1st_2nd_deriative():
    profile = np.ones(30)
    profile[0:5] = 6 # flat high
    profile[5:10] = np.linspace(5, 1, 5) # Ramp
    profile[14] = 2 # small noise
    # profile[18] = 10 # sharp noise
    profile[25:] = 6 # sharp step, then flat high

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../signal")
    from filters import padding
    hlPadSz = 1
    profilePad = padding(profile, hlPadSz)

    derivative_1st  = [profilePad[i+hlPadSz] - profilePad[i+hlPadSz-1]  for i in range(len(profile))]
    derivative_2nd  = [profilePad[i+hlPadSz+1] + profilePad[i+hlPadSz-1] - 2*profilePad[i+hlPadSz] for i in range(len(profile))]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    profile = profile[hlPadSz:-hlPadSz]
    derivative_1st  = derivative_1st [hlPadSz:-hlPadSz]
    derivative_2nd  = derivative_2nd [hlPadSz:-hlPadSz]
    xx = range(1, len(profile)+1)
    ax.plot(xx, profile, 'k-s', label='profile')
    ax.plot(xx, derivative_1st , 'o--', label='derivative_1st ')
    ax.plot(xx, derivative_2nd , 'g-.', label='derivative_2nd ')

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
    from PlotConfig import addLegend
    addLegend([ax])

def try_Combined_Enhance():
    '''Bone Combining Spatial Enhancement
    Sobel*G masked HEF
    '''
    IMFILE = os.path.join(DIPPATH, r'Fig0343(a)(skeleton_orig).tif')
    im = cv2.imread(IMFILE, 0)

    fltShape = (3, 3)

    # Laplace
    flt_L = LaplaceFilter3()
    imL = fftconvolve(im, flt_L)
    imL_cv = cv2.Laplacian(im, cv2.CV_32F)
    print(imL.dtype, np.percentile(imL, np.linspace(0, 100, 6)), sep='\n')
    print(imL_cv.dtype, np.percentile(imL_cv, np.linspace(0, 100, 6)), sep='\n')

    # Sobel py
    flt_sX = SobelFilter(fltShape)
    imdX = fftconvolve(im, flt_sX)
    flt_sY = SobelFilter(fltShape, 1)
    imdY = fftconvolve(im, flt_sY)
    im_Sobel = np.sqrt(imdX**2, imdY**2)
    pyimages = [imdX, imdY, im_Sobel, imL]
    print(imdX.dtype, np.percentile(imdX, np.linspace(0, 100, 6)), sep='\n')

    # Sobel cv
    imdX_cv = cv2.Sobel(im, cv2.CV_32F, 1, 0)
    imdY_cv = cv2.Sobel(im, cv2.CV_32F, 0, 1) #cv2.CV_8U
    im_Sobel_cv = np.sqrt(imdX_cv**2, imdY_cv**2)
    cvimages = [imdX_cv, imdY_cv, im_Sobel_cv, imL_cv]
    diffs = list(np.array(pyimages) - np.array(cvimages))
    print(imdX_cv.dtype, np.percentile(imdX_cv, np.linspace(0, 100, 6)), sep='\n')

    # Laplace enhanced
    imLE = im + imL
    imLE = np.clip(imLE, 0, 255)

    # Sobel*G x Laplace enhanced
    im_Sobel = np.abs(imdX) + np.abs(imdY)
    im_Sobel = np.clip(im_Sobel, 0, 255)
    # im_Sobel_G = fftconvolve(im_Sobel, GaussianFilter((5, 5) ) )
    im_Sobel_G = fftconvolve(im_Sobel, BoxFilter((5, 5) ) )
    im_Sobel_G = np.clip(im_Sobel_G, 0, 255)
    #shist = hist_lines(normalize(im_Sobel_G))
    mask = im_Sobel_G.astype(np.int32)*imLE.astype(np.int32)
    print(mask.dtype, np.percentile(mask, np.linspace(0, 100, 6)), sep='\n')
    mask = normalize(mask)
    print(mask.dtype, np.percentile(mask, np.linspace(0, 100, 6)), sep='\n')

    # Final enhanced = im + Sobel*G x Laplace enhanced
    imShapened = im + mask
    imShapened = np.clip(imShapened, 0, 255)
    # imShapenedPower = intensityTransform(imShapened, lambda x: x**0.5) #wrong
    imShapenedPower = intensityTransform(imShapened, powerFunc(1, 0.5))
    imshowMultiple([im, normalize(imL), normalize(imLE), normalize(im_Sobel)] + [normalize(im_Sobel_G), mask, imShapened, imShapenedPower],
        ['raw', 'Laplace 3x3', '(1+L)I', 'Sobel'] + [ 'Sobel*G',  '(1+L)I*Sobel_G mask', 'I + (1+L)I*Sobel_G', 'Fig 7 gamma=0.5'], **KWARGS
        )

    # imshowMultiple([im, imLE, mask, imShapened],
    #     ['raw'] + ['(1+L)I', '(1+L)I*Sobel_G mask', 'I + (1+L)I*Sobel_G'],
    #     **KWARGS)

    # imshowMultiple_TitleMatrix(pyimages + cvimages + diffs,
    #     3, 4,
    #     ['spatial', 'cv', 'diff'], ['dx', 'dy', 'Sobel', 'Laplace'],
    #     **KWARGS)

def try_Combined_Enhance_Ops():
    # use the newly add image enhancement operators

    # IMFILE = os.path.join(DIPPATH, r'Fig0343(a)(skeleton_orig).tif')
    IMFILE = os.path.join(WORKDIR, r'Calaveras_v3_p1521_LDose.bmp')
    im = cv2.imread(IMFILE, 0)
    #im=TrimmedMean(im, ksize=5, d=8)

    fltShape = (3, 3)

    # Laplace
    flt_L = LaplaceFilter3()
    imL = fftconvolve(im, flt_L)
    imLE = imAdd(im, imL, 255)

    # Sobel py
    flt_sX = SobelFilter(fltShape, 'x')
    imdX = fftconvolve(im, flt_sX)
    flt_sY = SobelFilter(fltShape, 'y')
    imdY = fftconvolve(im, flt_sY)
    im_Sobel = np.sqrt(imdX**2, imdY**2)
    im_Sobel = normalize(im_Sobel, Imax=255)
    #im_Sobel = imAdd(np.abs(imdX), np.abs(imdY), 255)

    # Sobel*G x Laplace enhanced
    im_Sobel_G = fftconvolve(im_Sobel, BoxFilter((5, 5) ) )
    im_Sobel_G = normalize(im_Sobel_G, 255)
    sharp = imMul(imLE, im_Sobel_G, 255)
    imShapened = imAdd(im, sharp, 255)
    dirname, filename, filextn = splitFileName(IMFILE)
    imShapenedMed = cv2.medianBlur(imShapened, 5)
    outfile = os.path.join(dirname, filename+'_shappen_med5x.'+filextn)
    cv2.imwrite(outfile, normalize(imShapenedMed, dtype=np.uint8))

    # imShapenedPower = intensityTransform(imShapened, powerFunc(1, 0.5))

    imshowMultiple([im, normalize(imL), normalize(imLE), normalize(im_Sobel)] + [normalize(im_Sobel_G), sharp, imShapened, imShapenedMed],
        ['raw', 'Laplace 3x3', '(1+L)I', 'Sobel'] + [ 'Sobel*G',  '(1+L)I*Sobel_G sharp mask', 'I + (1+L)I*Sobel_G', 'Fig 7 median 5x'], **KWARGS
        ) # 'Fig 7 gamma=0.5'

def main():
    import cProfile

    # try_equalizeHisto()

    # try_equalizeHisto_LowDose()

    # try_specifyHisto()

    # try_localHistoEq()
    # cProfile.run('try_localHistoEq()')

    # cProfile.run('try_Laplace_LoG()')
    try_Laplace_LoG()

    # try_1st_2nd_deriative()

    # try_Combined_Enhance()

    # try_Combined_Enhance_Ops()

if __name__ == '__main__':
    main()


