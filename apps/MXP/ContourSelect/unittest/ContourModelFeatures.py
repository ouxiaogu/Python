
# coding: utf-8

# In[43]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import sys
sys.path.insert(0, os.getcwd()+"/../../../../libs/common")
from FileUtil import gpfs2WinPath

caldatafile = r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/IMEC/Case02_calaveras_v3/3Tmp/CT_KPI_test/Calaveras_v3_regular_CT_KPI_003_slope_modified_revert_all_patterns/h/cache/dummydb/result/MXP/job1/ContourSelectModelCalibration430result1/caldata.txt'
caldatafile = gpfs2WinPath(caldatafile)

df = pd.read_csv(caldatafile, sep='\s+')

tgtColName = 'UserLabel'
neighborColNames = ['NeighborOrientation', 'NeighborParalism']#, 'NeighborParalism', 'NeighborContinuity',
allColNames = ['slope', 'intensity', 'ridge_intensity', 'contrast', 'EigenRatio', 'NeighborOrientation', 'NeighborParalism']
srcColNames = ['slope', 'intensity', 'ridge_intensity', 'NeighborOrientation', 'NeighborParalism']
tgtColName = 'UserLabel'

# In[19]:


df.loc[:, srcColNames].describe()

# In[44]:

def normalize_features():
    X = df.loc[:, srcColNames].values
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    print(Xmin, Xmax)
    df = df.transform(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    X = df.loc[:, srcColNames].values
    y = df.loc[:, tgtColName].values
    return X, y, Xmax, xmin

# In[5]:

get_ipython().magic(u'matplotlib auto')

def feature_hist(df):
    def sephist(col):
        TP = df[df[tgtColName] == 0][col]
        TN = df[df[tgtColName] == 1][col]
        return TP, TN
    labels = ['intensity', 'slope', 'ridge_intensity', 'contrast', 'EigenRatio']
    df.loc[:, 'EigenRatio'] = df.loc[:, 'EigenRatio'].abs()
    for num, alpha in enumerate(neighborColNames+labels):
        plt.subplot(2, 4, num+1)

        TP, TN = sephist(alpha)
        plt.hist((TP, TN), bins=25, alpha=0.5, label=map(''.join, zip(2*[tgtColName], 2*['=='], ['0', '1'])), color=['b', 'g'])
        #plt.hist(TP, bins=50, alpha=0.5, label=tgtColName+'==0', color='b')
        #plt.hist(TN, bins=50, alpha=0.5, label=tgtColName+'==1', color='g')
        plt.legend(loc='upper right')
        plt.title(alpha)
        plt.yscale('log')
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

# In[6]:

df.loc[:, tgtColName].value_counts()


# In[7]:


df.loc[df[tgtColName]==0, srcColNames].describe()


# In[8]:


df.loc[df[tgtColName]==1, srcColNames].describe()


# In[45]:
def svm(X, y)
    from sklearn import svm
    clf = svm.SVC(kernel='linear', class_weight='balanced') # {0: 10, 1: 1}
    model = clf.fit(X, y)

    modelform = pd.DataFrame(data=clf.coef_.flatten(), index=srcColNames)
    modelform.loc['intercept', 0] = clf.intercept_
    print(modelform)
    return model

# In[10]:

def DT(X, y)
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    model = clf.fit(X, y)
    return model

# In[46]:

def metric(X, y, model):
    from sklearn.metrics import confusion_matrix
    calcRMS = lambda y_pred, y: np.sqrt(np.mean(np.power(y_pred - y, 2)))
    calcMSE = lambda y_pred, y: np.mean(np.power(y_pred - y, 2))
    y_cal_pred = model.predict(X)
    rms = calcRMS(y_cal_pred, y)
    cm = confusion_matrix(y, y_cal_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Clf model on calibration set, RMS {} MSE {}".format(rms, calcMSE(y_cal_pred, y)))
    print("Clf model confusion matrix on calibration set:\n{}\n{}".format(cm, cm_norm))

# In[47]:

def predict(model, Xmin, Xmax):
    import sys
    import os.path
    sys.path.insert(0, os.getcwd()+"/../../../../libs/tacx")
    print(os.getcwd()+"/../../../../libs/tacx")
    from SEMContour import *
    sys.path.insert(0, os.getcwd()+"/../../../../libs/common")
    from FileUtil import gpfs2WinPath

    contourfile = r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/IMEC/Case02_calaveras_v3/3Tmp/CT_KPI_test/Calaveras_v3_regular_CT_KPI_003_slope_modified_revert_all_patterns/h/cache/dummydb/result/MXP/job1/ContourSelectModelCalibration430result1/461_image_contour.txt'
    contourfile = gpfs2WinPath(contourfile)

    class ContourAnalyzer(object):
        """docstring for ContourData"""
        def __init__(self, contourfile):
            self.__build(contourfile)

        def __build(self, contourfile):
            contour = SEMContour()
            contour.parseFile(contourfile)
            if not contour:
                sys.exit("ERROR: read in contour file %s fails\n" % contourfile)
            self.contour = contour
            self.df = contour.toDf()
    # get contour data
    ca = ContourAnalyzer(contourfile)
    contour = ca.contour
    df = ca.df


    # In[17]:


    df.loc[:, srcColNames].describe()


    # In[48]:


    get_ipython().magic(u'matplotlib auto')

    X_test = df.loc[:, srcColNames].values
    X_test = np.array([(X_test[:,i] - Xmin[i])/(Xmax[i] - Xmin[i]) for i in range(len(srcColNames)) ]).T
    df.loc[:, 'ClfLabel'] = model.predict(X_test)
    # SEM Contour Selection resulst plot: by classifer Positive 0, & Negative 1
    def plotContourDiscriminator(contour, im=None, wndname=''):
        # plot image and classified contour point
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

        df = contour.toDf()
        Positive = df.ClfLabel==0
        Negative = df.ClfLabel==1

        # calculate confusion matrix
        cm = np.array([len(df.loc[flt, :]) for flt in [Positive, Negative]])
        cm_norm = cm.astype('float') / cm.sum()
        
        if im is not None:
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
    plotContourDiscriminator(contour.fromDf(df))

