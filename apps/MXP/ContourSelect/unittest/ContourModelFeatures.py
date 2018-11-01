
# coding: utf-8

# In[1]:


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
allColNames = ['NeighborContinuity', 'NeighborOrientation', 'NeighborParalism', 'slope', 'intensity', 'ridge_intensity', 'contrast', 'EigenRatio']
srcColNames = ['slope', 'intensity', 'ridge_intensity', 'NeighborOrientation', 'NeighborParalism']
tgtColName = 'UserLabel'


# In[10]:


wiScaling = 1
scaling = lambda X_Arr: np.array([(X_Arr[i] - Xmin[i])/(Xmax[i] - Xmin[i]) for i in range(len(srcColNames)) ]).T
if wiScaling:
    X_cal = df.loc[df.usage=='CAL', srcColNames].values
    Xmin = X_cal.min(axis=0)
    Xmax = X_cal.max(axis=0)
    print(Xmin, Xmax)
    df.loc[:, srcColNames] = df.loc[:, srcColNames].apply(scaling, axis=1)

X_cal = df.loc[df.usage=='CAL', srcColNames].values
y_cal = df.loc[df.usage=='CAL', tgtColName].values
X_ver = df.loc[df.usage=='VER', srcColNames].values
y_ver = df.loc[df.usage=='VER', tgtColName].values


# In[11]:


df.loc[:, srcColNames].describe()


# In[ ]:


get_ipython().magic('matplotlib auto')

# hist plot 

df2 = pd.melt(df, id_vars=tgtColName, value_vars=neighborColNames, value_name='value')
bins=np.linspace(df2.value.min(), df2.value.max(), 100)
g = sns.FacetGrid(df2, col="variable", hue=tgtColName, palette="Set1", col_wrap=3)
g.map(plt.hist, 'value', bins=bins, ec="k")
plt.yscale('log')
g.axes[-1].legend()

# scatter plot 
#sns.pointplot('NeighborOrientation', 'NeighborParalism', hue=tgtColName, data=df)

plt.show()


# In[21]:


get_ipython().magic('matplotlib auto')
def sephist(col):
    TP = df[df[tgtColName] == 0][col]
    TN = df[df[tgtColName] == 1][col]
    return TP, TN
#df.loc[:, 'slope'] = df.loc[:, 'slope'].abs()
for num, alpha in enumerate(allColNames):
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


# In[ ]:


df.loc[:, tgtColName].value_counts()


# In[ ]:


df.loc[df[tgtColName]==0, srcColNames].describe()


# In[ ]:


df.loc[df[tgtColName]==1, srcColNames].describe()


# In[22]:


from sklearn import svm
clf = svm.SVC(kernel='linear', class_weight='balanced') # {0: 10, 1: 1}
model = clf.fit(X_cal, y_cal)

modelform = pd.DataFrame(data=clf.coef_.flatten(), index=srcColNames)
modelform.loc['intercept', 0] = clf.intercept_
print(modelform)


# In[26]:


from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=0) # , max_depth=len(srcColNames)+1, min_samples_split=3
model = clf.fit(X_cal, y_cal)
feature_importance = pd.DataFrame(data=model.feature_importances_.flatten(), index=srcColNames)
print(feature_importance)

#print(model.decision_path(X_cal))
print(model.get_params())


# In[19]:


import graphviz 
dot_data = tree.export_graphviz(model, out_file=None, 
                         feature_names=srcColNames,  
                         class_names=tgtColName,  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data)  
graph


# In[23]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, random_state=0)
model = clf.fit(X_cal, y_cal)
feature_importance = pd.DataFrame(data=model.feature_importances_.flatten(), index=srcColNames)
print(feature_importance)


# In[17]:


from sklearn.metrics import confusion_matrix
calcRMS = lambda y_pred, y: np.sqrt(np.mean(np.power(y_pred - y, 2)))
def predict(X, y, usage='CAL'):
    y_pred = model.predict(X)
    rms = calcRMS(y_pred, y)
    cm = confusion_matrix(y, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Clf model rms on {} set: {}".format(usage, rms))
    print("Clf model confusion matrix on {} set:\n{}\n{}".format(usage, cm, cm_norm))
predict(X_cal, y_cal)
predict(X_ver, y_ver, 'VER')


# In[14]:


get_ipython().magic('matplotlib auto')
import sys
import os.path
sys.path.insert(0, os.getcwd()+"/../../../../libs/tacx")
print(os.getcwd()+"/../../../../libs/tacx")
from SEMContour import *
sys.path.insert(0, os.getcwd()+"/../../../../libs/common")
from FileUtil import gpfs2WinPath

import glob

CWD = ''.join(['/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/IMEC/Case02_calaveras_v3/3Tmp/CT_KPI_test/Calaveras_v3_regular_CT_KPI_003_slope_modified_revert_all_patterns/'
      'h/cache/dummydb/result/MXP/job1/ContourSelectModelCalibration430result1'])

''' # comment block 1 starts
#################
# type 1, review model apply image by random permutation
#################
pathfilter = '*_image_contour.txt'
pathex = gpfs2WinPath(os.path.join(CWD, pathfilter))
contourfiles = glob.glob(pathex)
contourindice = np.random.permutation(np.arange(len(contourfiles)))
for ii in range(0*8, 1*8):
    fig = plt.figure()
    for jj, idx in enumerate(contourindice[ii*8:(ii+1)*8]):
        contourfile = contourfiles[idx]
        patternid = os.path.basename(contourfile).strip('_image_contour.txt')
        ################# end of type 1
''' # comment block 1 ends
        
#################
# type 2, review model apply image by giving list
#################
patternids = [461, 1001]

for ii in range(int(np.ceil(len(patternids)/8.))):
    fig = plt.figure()
    for jj, idx in enumerate(range(ii*8, (ii+1)*8)):
        patternid = str(patternids[idx])
        contourfile = gpfs2WinPath(os.path.join(CWD, patternid+'_image_contour.txt'))
        ################# end of type 2        
        
        
        if not os.path.exists(contourfile):
            print(patternid+' not exist')
            continue

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


        X_test = df.loc[:, srcColNames].values
        X_test = np.array([(X_test[:,i] - Xmin[i])/(Xmax[i] - Xmin[i]) for i in range(len(srcColNames)) ]).T
        df.loc[:, 'ClfLabel'] = model.predict(X_test)
        # SEM Contour Selection resulst plot: by classifer Positive 0, & Negative 1
        def plotContourDiscriminator(contour, im=None, wndname=''):
            # plot image and classified contour point
            
            ax = fig.add_subplot(2,4,jj+1)

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
                    linestyle='None', marker= 'o', markeredgecolor='r', markersize=2, markeredgewidth=1, markerfacecolor='none', 
                    label='remove: {}({:.3f}%)'.format(cm[0], cm_norm[0]*100 )) #Discriminator Positive, ClfLabel=0
            ax.plot(df.loc[Negative ,'offsetx'], df.loc[Negative, 'offsety'], #'r*', markersize=2,
                    linestyle='None', marker= '.', markeredgecolor='b', markersize=2, markeredgewidth=1, markerfacecolor='none', 
                    label='Keep: {}({:.3f}%)'.format(cm[1], cm_norm[1]*100 )) #Discriminator Negative, ClfLabel=1:

            #ax = plt.gca() # gca() function returns the current Axes instance
            #ax.set_ylim(ax.get_ylim()[::-1]) # reverse Y
            plt.gca().invert_yaxis()
            plt.legend(loc=1)
            plt.show()
        plotContourDiscriminator(contour.fromDf(df), wndname='Pattern '+ patternid)

