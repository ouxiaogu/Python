# %matplotlib auto
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2

import sys
import os.path
sys.path.insert(0, os.getcwd()+"/../../tacx")
from SEMContour import *
from MxpStage import MxpStageXmlParser
sys.path.insert(0, os.getcwd()+"/../../imutil")
from ImGUI import imread_gray
sys.path.insert(0, os.getcwd()+"/../../common")
from PlotConfig import *
from FileUtil import gpfs2WinPath

CWD = r'C:\Localdata\D\Note\Python\apps\MXP\ContourSelect\samplejob\h\cache\dummydb\result\MXP\job1'
CWD = gpfs2WinPath(CWD)

inxml = os.path.join(CWD, r'contourselcal430out.xml')

def loadPatternData(imgfile='', contourfile=''):
    bSucceedReadCt, bSucceedReadIm = False, False

    # read contour
    contour = SEMContour()
    bSucceedReadCt = contour.parseFile(contourfile)
    if not bSucceedReadCt:
        raise OSError("Error, contourfile('{}') cannot be parsed".format(contourfile))
        
    im = None
    try:  # read image
        im, _ = imread_gray(imgfile)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    except:
        # raise
        pass
    return im, contour

def plotContourClassifier(inxml):
    icf_parser = MxpStageXmlParser(inxml)
    # icf = icf_parser.icf
    df_icf = icf_parser.iccfs2df()
    patternid = 461
    df_iccf = df_icf.loc[df_icf.name==patternid, :]
    sr_iccf = pd.Series
    contourfile = df_iccf.loc['contour/path']
    contourfile = CWD + '/' + contourfile
    imgfile = df_iccf.loc['image/path']
    imgfile = CWD + '/' + imgfile
    im, contour = loadPatternData(imgfile, contourfile)

    # TODO, add bbox plot
    # df.filter(regex='bbox/outlier', axis=1)

    # plot image and classified contour point
    fig = plt.figure()
    ax = fig.add_subplot(111)
    imw, imh = contour.getshape()
    ax.set_xlim([0, imw])
    ax.set_ylim([0, imh])
    ax.set_title('Pattern '+str(patternid))

    df = contour.todf()
    UserGood = df.UserLabel == 1
    FalseClf = df.UserLabel != df.clf_label
    ax.imshow(im)
    ax.plot(df.loc[UserGood ,'offsetx'], df.loc[UserGood, 'offsety'], 'b.', markersize=1, label='User Label as good')
    ax.plot(df.loc[~UserGood ,'offsetx'], df.loc[~UserGood, 'offsety'], 'r.', markersize=3, label='User Label as bad')
    ax.plot(df.loc[FalseClf ,'offsetx'], df.loc[FalseClf, 'offsety'], marker= 'o', markersize=4, markeredgewidth=1, label='Classifier differ with User Label')
    plt.legend()
    plt.show()


plotContourClassifier(inxml)