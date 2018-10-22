
# coding: utf-8

# In[1]:


# %matplotlib auto
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2

import sys
import os.path
print(os.getcwd())
sys.path.insert(0, os.getcwd()+"/../../../../libs/tacx")
print(os.getcwd()+"/../../../../libs/tacx")
from SEMContour import *
from MxpStage import MxpStageXmlParser
sys.path.insert(0, os.getcwd()+"/../../../../libs/imutil")
from ImGUI import imread_gray
sys.path.insert(0, os.getcwd()+"/../../../../libs/common")
from PlotConfig import *
from FileUtil import gpfs2WinPath

# CWD = r'C:\Localdata\D\Note\Python\apps\MXP\ContourSelect\samplejob\h\cache\dummydb\result\MXP\job1'
CWD = r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/IMEC/Case02_calaveras_v3/3Tmp/CT_KPI_test/Calaveras_v3_regular_CT_KPI_003_slope_modified_revert_all_patterns/h/cache/dummydb/result/MXP/job1/'
CWD = gpfs2WinPath(CWD)

inxml = os.path.join(CWD, r'contourselcal430out.xml') # contourextraction400out.xml contourlabeling410out.xml, contourselcal430out.xml

def loadPatternData(imgfile='', contourfile=''):
    bSucceedReadCt = False

    # read contour
    contour = SEMContour()
    bSucceedReadCt = contour.parseFile(contourfile)
    if not bSucceedReadCt:
        raise OSError("Error, contourfile('{}') cannot be parsed".format(contourfile))
        
    im = None
    try:  # read image
        im, _ = imread_gray(imgfile)
        im = (im/65535).astype(np.float32)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    except:
        # raise
        pass
    return im, contour

def getContourClassifierData(inxml):
    print(inxml)
    ocf_parser = MxpStageXmlParser(inxml) #'inxml', 'outxml'
    # icf = icf_parser.icf
    df_ocf = ocf_parser.occfs2df()
    patternid = 461
    df_occf = df_ocf.loc[df_ocf.name==patternid, :]
    sr_occf = pd.Series(df_occf.values.flatten(), index=df_occf.columns)
    contourfile = getRealFilePath(sr_occf.loc['contour/path'])
    imgfile = getRealFilePath(sr_occf.loc['image/path'])
    im, contour = loadPatternData(imgfile, contourfile)
    return im, contour
    # TODO, add bbox plot
    # df.filter(regex='bbox/outlier', axis=1)


# In[ ]:


# plot by column unique labels
def plot_col_by_label(contour, patternid='', colname=''):
    df = contour.toDf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    xini, yini, xend, yend = contour.getBBox()
    ax.set_xlim([xini, xend])
    ax.set_ylim([yini, yend])
    ax.set_title("Pattern "+patternid)
    
    uniqVals = df.loc[:, colname].drop_duplicates().values
    print(uniqVals)
    for label in uniqVals:
        flt_eq = df.loc[:, colname] == label
        if label == 'nan':
            flt_eq = df.loc[:, colname].isna()
        ax.plot(df.loc[flt_eq, 'offsetx'], df.loc[flt_eq, 'offsety'], '.', 
                linestyle='None',  markersize=2, label=colname+'=={}'.format(label))

    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()


# In[ ]:


# plot the SEM image, contour and angle
def plot_image_contour_angle(im, contour, patternid='', arrow_length=1):
    df = contour.toDf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    #imw, imh = contour.getshape()
    #ax.set_xlim([0, imw])
    #ax.set_ylim([0, imh])
    xini, yini, xend, yend = contour.getBBox()
    ax.set_xlim([xini, xend])
    ax.set_ylim([yini, yend])
    ax.set_title("Pattern "+patternid+ " image Contour")
    
    # plot image
    ax.imshow(im)
    
    # plot contour
    ax.plot(df.loc[:, 'offsetx'], df.loc[:, 'offsety'], 'b.')
    ax.plot(250.480209, 715.985352, 'r.')
    
    # plot angle
    for _, row in df.iterrows():
        x, y = row.loc['offsetx'], row.loc['offsety']
        angle = row.loc['angle']
        dx, dy = arrow_length*np.cos(angle), arrow_length*np.sin(angle)
        ax.arrow(x, y, dx, dy, width=0.1, fc='y', ec='y') # ,shape='right', overhang=0
        
    plt.gca().invert_yaxis()
    plt.show()


# In[ ]:


# SEM Contour Selection resulst plot: by TP, FN, FP, TP
def plotContourClassifier(im, contour, wndname=''):
    # plot image and classified contour point
    df = contour.toDf()
    if any([col not in df.columns for col in ['UserLabel', 'ClfLabel']]):
        return False

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    imw, imh = contour.getshape()
    ax.set_aspect('equal')
    '''
    ax.set_xlim([0, imw])
    ax.set_ylim([0, imh])
    '''
    xini, yini, xend, yend = contour.getBBox()
    ax.set_xlim([xini, xend])
    ax.set_ylim([yini, yend])
    ax.set_title(wndname)

    TP = (df.UserLabel==0) & (df.ClfLabel==0)
    FN = (df.UserLabel==0) & (df.ClfLabel==1)
    FP = (df.UserLabel==1) & (df.ClfLabel==0)
    TN = (df.UserLabel==1) & (df.ClfLabel==1)
    
    # calculate confusion matrix
    cm = np.array([len(df.loc[flt, :]) for flt in [TP, FN, FP, TN]]).reshape((2, 2))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    ax.imshow(im)
    ax.plot(df.loc[TP ,'offsetx'], df.loc[TP, 'offsety'], #'b.', markersize=1, 
            linestyle='None', marker= 'o', markersize=2, markeredgewidth=1, markerfacecolor='none', 
            label='TP, UserLabel=0 & ClfLabel=0: {}({:.3f}%)'.format(cm[0, 0], cm_norm[0, 0]*100 ))
    ax.plot(df.loc[FN ,'offsetx'], df.loc[FN, 'offsety'],
            linestyle='None', marker= 'o', markersize=4, markeredgewidth=1, markerfacecolor='none', 
            label='FN, UserLabel=1 & ClfLabel=0: {}({:.3f}%)'.format(cm[0, 1], cm_norm[0, 1]*100 ))
    ax.plot(df.loc[FP ,'offsetx'], df.loc[FP, 'offsety'], 
            linestyle='None', marker= 'o', markersize=4, markeredgewidth=1, markerfacecolor='none', 
            label='FP, UserLabel=0 & ClfLabel=1: {}({:.3f}%)'.format(cm[1, 0], cm_norm[1, 0]*100 ))
    ax.plot(df.loc[TN ,'offsetx'], df.loc[TN, 'offsety'], #'r*', markersize=2,
            linestyle='None', marker= 'o', markersize=2, markeredgewidth=1, markerfacecolor='none', 
            label='TN, UserLabel=1 & ClfLabel=1: {}({:.3f}%)'.format(cm[1, 1], cm_norm[1, 1]*100 ))
    
    #ax = plt.gca() # gca() function returns the current Axes instance
    #ax.set_ylim(ax.get_ylim()[::-1]) # reverse Y
    plt.gca().invert_yaxis()
    plt.legend(loc=1)
    plt.show()
    return True
    
# SEM Contour Selection resulst plot: by classifer Positive 0, & Negative 1
def plotContourDiscriminator(im, contour, wndname=''):
    # plot image and classified contour point
    df = contour.toDf()
    if any([col not in df.columns for col in ['ClfLabel']]):
        return False

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    imw, imh = contour.getshape()
    '''
    ax.set_aspect('equal')
    ax.set_xlim([0, imw])
    ax.set_ylim([0, imh])
    '''
    xini, yini, xend, yend = contour.getBBox()
    ax.set_xlim([xini, xend])
    ax.set_ylim([yini, yend])
    ax.set_title(wndname)
    
    Positive = df.ClfLabel==0
    Negative = df.ClfLabel==1

    # calculate confusion matrix
    cm = np.array([len(df.loc[flt, :]) for flt in [Positive, Negative]])
    cm_norm = cm.astype('float') / cm.sum()
    
    ax.imshow(im)
    ax.plot(df.loc[Positive ,'offsetx'], df.loc[Positive, 'offsety'], #'b.', markersize=1, 
            linestyle='None', marker= 'o', markersize=2, markeredgewidth=1, markerfacecolor='none', 
            label='Discriminator Positive, ClfLabel=0: {}({:.3f}%)'.format(cm[0], cm_norm[0]*100 ))
    ax.plot(df.loc[Negative ,'offsetx'], df.loc[Negative, 'offsety'], #'r*', markersize=2,
            linestyle='None', marker= 'o', markersize=2, markeredgewidth=1, markerfacecolor='none', 
            label='Discriminator Negative, ClfLabel=1: {}({:.3f}%)'.format(cm[1], cm_norm[1]*100 ))
    
    #ax = plt.gca() # gca() function returns the current Axes instance
    #ax.set_ylim(ax.get_ylim()[::-1]) # reverse Y
    plt.gca().invert_yaxis()
    plt.legend(loc=1)
    plt.show()
    return True

# In[ ]:


def plotAllContourClfData(inxml):
    ocf_parser = MxpStageXmlParser(inxml) #, 'outxml'
    # icf = icf_parser.icf
    df_ocf = ocf_parser.occfs2df()
    
    # samples
    df_ocf = df_ocf.loc[(df_ocf['name'] > 243) & (df_ocf['costwt'] > 0), :]
    names = df_ocf.loc[:, 'name'].values
    np.random.seed(128)
    np.random.shuffle(names)
    names = names[0:20:2]

    for _, row in df_ocf.iterrows():
        patternid = row.loc['name']
        if patternid not in names:
            continue
        print("Start to process pattern {}".format(patternid))
        #usage = row.loc['usage']
        #print("{} {}".format(patternid, usage))
        
        contourfile = getRealFilePath(row.loc['contour/path'])
        imgfile = getRealFilePath(row.loc['image/path'])
        im, contour = loadPatternData(imgfile, contourfile)
        if plotContourClassifier(im, contour, "Pattern "+str(patternid)) or plotContourDiscriminator(im, contour, "Pattern "+str(patternid)):
            print("Successfully processed pattern {}".format(patternid))
        else:
            print("Can't plot pattern {}".format(patternid))

def getRealFilePath(curfile):
    if '/' in curfile:
        ossep = '/'
    else:
        ossep = '\\'
    return os.path.join(CWD, os.sep.join(curfile.split(ossep)))

def main():
    singlePatternPlot = 1
    if singlePatternPlot:
        im, contour = getContourClassifierData(inxml)
        #plot_col_by_label(contour, patternid='461', colname="UserLabel")
        #plot_image_contour_angle(im, contour, '461')

        plotContourClassifier(im, contour, 'Pattern 461')
        plotContourDiscriminator(im, contour, 'Pattern 461')
    else:
        plotAllContourClfData(inxml)

# In[ ]:

if __name__ == '__main__':
    main()



