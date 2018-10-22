
# coding: utf-8

# In[31]:


get_ipython().magic('matplotlib auto')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import sys
import os.path
sys.path.insert(0, os.getcwd()+"/..")
from SEMContour import *
sys.path.insert(0, os.getcwd()+"/../../common")
from PlotConfig import *
from FileUtil import gpfs2WinPath

#CWD = '/gpfs/WW/BD/MXP/SEM_IMAGE/IMEC/Case02_calaveras_v3/3Tmp/CT_KPI_test/Calaveras_v3_regular_CT_KPI_003_slope_modified_revert_all_patterns/h/cache/dummydb/result/MXP/job1/ContourSelectModelCalibration430result1'
CWD = r'C:\Localdata\D\Note\Python\apps\MXP\ContourSelect\samplejob\h\cache\dummydb\result\MXP\job1\ContourExtraction400result1'
#CWD = r'C:\Localdata\D\Note\Python\apps\MXP\ContourSelect\samplejob1\h\cache\dummydb\result\MXP\job1\ContourExtraction400result1'
CWD = gpfs2WinPath(CWD)
allNeighborColNames = ['NeighborContinuity', 'NeighborOrientation', 'NeighborParalism']


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
contourfile = os.path.join(CWD, '1001_image_contour.txt')
ca = ContourAnalyzer(contourfile)
df = ca.df


# In[ ]:


def plot_corr(df):
    matplotlib.style.use('ggplot')
    #plot_contour(self.contour)
    # cols = 'slope  ridge_intensity intensity  contrast'.split()
    cols = 'slope  ridge_intensity'.split()
    print(df.columns)
    df = df[cols]
    df.loc[:, 'slope'] = df.loc[:, 'slope'].abs().values
    
    from pandas.plotting import scatter_matrix
    colors = ['red','blue']
    scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde', color=colors) 
    
    '''
    import seaborn as sns
    sns.set(style="ticks")
    sns.pairplot(df, kind='scatter', diag_kind='kde')
    '''


# In[14]:


# plot the SEM contour and angle
def plot_contour_angle(ca, patternid='', arrow_length=1):
    df = ca.df
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    imw, imh = ca.contour.getshape()
    ax.set_xlim([0, imw])
    ax.set_ylim([0, imh])
    ax.set_title("Pattern "+patternid+ " image Contour")
    
    # plot image
    
    # plot contour
    ax.plot(df.loc[:, 'offsetx'], df.loc[:, 'offsety'], 'b.')
    
    # plot angle
    for _, row in df.iterrows():
        x, y = row.loc['offsetx'], row.loc['offsety']
        angle = row.loc['angle']
        dx, dy = arrow_length*np.cos(angle), arrow_length*np.sin(angle)
        ax.arrow(x, y, dx, dy, width=0.1, fc='y', ec='y') # ,shape='right', overhang=0
        
    plt.gca().invert_yaxis()
    plt.show()
plot_contour_angle(ca, '461')


# In[12]:


# plot the histgram for the modified slope, & plot by filter
print(df.columns)
colname = 'slope'
df[colname].plot.hist(bins=100)
def plot_col_filter(ca, patternid='', colname=''):
    df = ca.df
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    imw, imh = ca.contour.getshape()
    ax.set_xlim([0, imw])
    ax.set_ylim([0, imh])
    ax.set_title("Pattern "+patternid)
    
    thresh = 0
    flt_gt = df.loc[:, colname] > thresh
    flt_eq = df.loc[:, colname] == thresh
    flt_lt = df.loc[:, colname] < thresh
    
    ax.plot(df.loc[flt_gt, 'offsetx'], df.loc[flt_gt, 'offsety'], 'b.', markersize=2, label=colname+'>{}'.format(thresh))
    ax.plot(df.loc[flt_eq, 'offsetx'], df.loc[flt_eq, 'offsety'], 'r.', markersize=2, label=colname+'=={}'.format(thresh))
    ax.plot(df.loc[flt_lt, 'offsetx'], df.loc[flt_lt, 'offsety'], 'g.', markersize=2, label=colname+'<{}'.format(thresh))

    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
plot_col_filter(ca, patternid='461', colname=colname)


# In[33]:


def addNeighborFeatures(df):
    '''
    add Features for the input contour DataFrame, based on the neighbor relationship in the context of segment

    Parameters:
    -----------
    df: [in, out] contour as DataFrame
        [in] Contour df, must contains `polygonId`, `angle`, `offsetx`, `offsety`
        [out] Contour df, added `NeighborContinuity`, `NeighborOrientation`, `NeighborParalism`

            - `NeighborContinuity`:  |X(n) - X(n-1)|^2, usually is to 1 (because of 8-neighbor contour tracing)
            - `NeighborOrientation`:  dot(EigenVector(n), EigenVector(n-1)), closer to 1, the better(may use 1-dot)
            - `NeighborParalism`:  ||cross((X(n) - X(n-1)), EigenVector(n-1))||, closer to 1, the better(may use 1-cross)
    TODO, the segment neighborhood based features can only be obtained by the whole segment, can't use ROI cropped segment 
    '''
    if len(df) <= 0:
        return df
    polygonIds = df.loc[:, 'polygonId'].drop_duplicates().values
    preIdx = df.index[0]
    for polygonId in polygonIds:
        isPolygonHead = True
        for curIdx, _ in df.loc[df['polygonId']==polygonId, :].iterrows():
            NeighborContinuity = 1
            NeighborOrientation = 1
            NeighborParalism = 1
            if not isPolygonHead:
                eigenvector_n_1 = np.array([np.cos(df.loc[preIdx, 'angle']), np.sin(df.loc[preIdx, 'angle'])])
                eigenvector_n = np.array([np.cos(df.loc[curIdx, 'angle']), np.sin(df.loc[curIdx, 'angle'])])
                neighorvector = np.array([df.loc[curIdx, 'offsetx'] - df.loc[preIdx, 'offsetx'],
                                        df.loc[curIdx, 'offsety'] - df.loc[preIdx, 'offsety']])
                crossvector = np.cross(neighorvector, eigenvector_n_1)

                NeighborContinuity = np.sqrt(neighorvector.dot(neighorvector))
                NeighborOrientation = eigenvector_n.dot(eigenvector_n_1)
                NeighborParalism = np.sqrt(crossvector.dot(crossvector))/NeighborContinuity
                NeighborContinuity = NeighborContinuity
            preIdx = curIdx
            isPolygonHead = False

            for ii, val in enumerate([NeighborContinuity, NeighborOrientation, NeighborParalism]):
                colname = allNeighborColNames[ii]
                df.loc[curIdx, colname] = val
    return df

def plot_multi_filters(ca, patternid='', strFlts=None):
    if strFlts is None:
        strFlts = []
    df = ca.df
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    xini, yini, xend, yend = ca.contour.getBBox()
    ax.set_xlim([xini, xend])
    ax.set_ylim([yini, yend])
    ax.set_title("Pattern "+patternid)
    
    # plot contour
    ax.plot(df.loc[:, 'offsetx'], df.loc[:, 'offsety'], 'k.', markersize=1, label='SEM Contour')

    # plot filters
    for strFlt in strFlts:
        curdf = df.query(strFlt)
        ax.plot(curdf.loc[:, 'offsetx'], curdf.loc[:, 'offsety'], 'o', markersize=4, label=strFlt, alpha=0.6)

    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()

df = addNeighborFeatures(df)
plot_multi_filters(ca, patternid='1001', strFlts=['abs(1-NeighborContinuity) > 0.5', 'NeighborParalism<0.98', 'NeighborOrientation<0.98'])


# In[ ]:


# plot contour filtering by ridge_intensity 
def plot_rd_filter(ca, patternid=''):
    df = ca.df
    imw, imh = ca.contour.getshape()
    
    figw = 9
    fig = plt.figure(figsize=(figw, figw))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim([0, imw])
    ax.set_ylim([0, imh])

    plt.gca().invert_yaxis()
    thresh = 0.003
    flt = df.loc[:, 'ridge_intensity'] > thresh
    ax.plot(df.loc[flt, 'offsetx'], df.loc[flt, 'offsety'], 'g.', markersize=2, label='ridge_intensity>{}'.format(thresh))
    ax.plot(df.loc[~flt, 'offsetx'], df.loc[~flt, 'offsety'], 'r.', markersize=3, label='ridge_intensity<={}'.format(thresh))
    
    ax.set_title(patternid+" Rg>{} filter".format(thresh))
    plt.legend()
    plt.show()
plot_rd_filter(ca, 'Pattern 3658')


# In[ ]:


# for pop out plot
get_ipython().magic('matplotlib qt5')

def plot_reg(ca, winname=''):
    df = ca.df
    imw, imh = ca.contour.getshape()
    
    colstr = 'slope  ridge_intensity'
    cols = colstr.split()
    #df = df[cols]
    df.loc[:, 'slope'] = df.loc[:, 'slope'].abs().values
    ## df = df.loc[df.slope<0.03, :]
    x, y = df.loc[:, 'slope'], df.loc[:, 'ridge_intensity']

    # from scipy import stats
    # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    #plt.plot(x, intercept + slope*x, 'r', label='fitted ridge_intensity')
    import statsmodels.api as sm
    X = sm.add_constant(x, prepend=False)
    results = sm.OLS(y, X).fit()
    # print(results.summary())
    # print(results.mse_resid, results.mse_total)
    # print(results.params, type(results.params))
    k, b = results.params.loc['slope'], results.params.loc['const']
    
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
    pred_std, predict_ci_low, predict_ci_upp = wls_prediction_std(results)

    xmax, ymax = x.max(), y.max()
    figw = 7
    fig = plt.figure(figsize=(figw, figw*ymax/xmax))
    #fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, xmax*1.1])
    ax.set_ylim([0, ymax*1.1])
    
    ax.set_title(winname+' ridge_intensity v.s. abs of modified slope')
    ax.set_xlabel('modified slope')
    ax.set_ylabel('ridge_intensity')
    
    ax.plot(x, y, 'o', label='original ridge_intensity v.s. slope')
    y_pred = results.predict()
    ax.plot(x, y_pred, 'r', label='predicted ridge_intensity={:.3f}slope+{:.3f}, $R^2={:.3f}$'.format(k, b, results.rsquared))
    plt.plot(x, predict_ci_low, 'b--', lw=1, label='predict lower')
    plt.plot(x, predict_ci_upp, 'g--', lw=1, label='predict upper')
    
    df.loc[:, 'predict_ci_low'] = predict_ci_low
    df.loc[:, 'predict_ci_upp'] = predict_ci_upp
    
    plt.legend()
    #plt.show()
    
    ## ridge_intensity prediction boundary plot
    figw = 9
    fig = plt.figure(figsize=(2*figw, figw))
    ax = fig.add_subplot(1, 2, 1)
    #fig = plt.figure(2)
    #ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim([0, imw])
    ax.set_ylim([0, imh])
    plt.gca().invert_yaxis()
    #ax.set_ylim(ax.get_ylim()[::-1])
    
    # print(df.columns)
    flt = (df.ridge_intensity>=df.predict_ci_low)& (df.ridge_intensity<=df.predict_ci_upp)
    nonzero = df.slope != 0
    #ax.plot(df.loc[flt ,'offsetx'], 1024-1-df.loc[flt, 'offsety'], 'b.', markersize=3, label='ridge_intensity In prediction range')
    ax.plot(df.loc[flt ,'offsetx'], df.loc[flt, 'offsety'], 'b.', markersize=2, label='ridge_intensity In prediction range')
    ax.plot(df.loc[(~flt)&nonzero ,'offsetx'], df.loc[(~flt)&nonzero, 'offsety'], 'co', markersize=5, label='Rg Out prediction range, slope!=0')
    ax.plot(df.loc[(~flt)&(~nonzero) ,'offsetx'], df.loc[(~flt)&(~nonzero), 'offsety'], 'rd', markersize=5, label='Rg Out prediction range, slope==0')
    ax.set_title(winname+" Rg outside prediction range of $Rg={:.3f}slope+{:.3f}$".format(k, b))
    ax.legend()
    
    ## ridge_intensity > thresh filter plot
    ax = fig.add_subplot(1, 2, 2)
    ax.set_aspect('equal')
    ax.set_xlim([0, imw])
    ax.set_ylim([0, imh])

    plt.gca().invert_yaxis()
    thresh = 0.003
    flt = df.loc[:, 'ridge_intensity'] > thresh
    ax.plot(df.loc[flt, 'offsetx'], df.loc[flt, 'offsety'], 'g.', markersize=2, label='ridge_intensity>{}'.format(thresh))
    ax.plot(df.loc[~flt, 'offsetx'], df.loc[~flt, 'offsety'], 'r.', markersize=3, label='ridge_intensity<={}'.format(thresh))
    
    ax.set_title(winname+" Rg>{} filter".format(thresh))
    ax.legend()
    
    
    plt.show()
    
    #resid=y-y_pred
    #rss=np.sum(resid**2)
    #MSE=np.sqrt(rss/(result.nobs-2))
    
    def ols_quantile(m, X, q):
      # m: Statsmodels OLS model.
      # X: X matrix of data to predict.
      # q: Quantile.
      #
      from scipy.stats import norm
      mean_pred = m.predict(X)
      se = np.sqrt(m.scale)
      return mean_pred + norm.ppf(q) * se
    
    #print(ols_quantile(results, X, 0.5))
    return results

    
results = plot_reg(ca)
print("ridge_intensity v.s. slope regression results:")
print(results.summary())
print('\nresults.mse\n', results.mse_resid, results.mse_total, '\n')
print("results.params\n", results.params, type(results.params))


# In[ ]:


patterns = [1, 444, 461, 1001, 3658]
contourfiles= [CWD+'/{}_image_contour.txt'.format(pid) for pid in patterns]


# 
# - for slightly better for visualazation
# 
#     %matplotlib notebook 
# 
# - normal
# 
#     %matplotlib inline 

# In[ ]:


get_ipython().magic('matplotlib qt5')
### ridge_intensity v.s. slope regression plot for more patterns
for ix, contourfile in enumerate(contourfiles):
    ca = ContourAnalyzer(contourfile)
    iminfo = 'Pattern '+str(patterns[ix])
    plot_reg(ca, iminfo)
    #plot_rd_filter(ca, iminfo)


# In[ ]:


get_ipython().magic('matplotlib auto')
def plot_reg2(df):
    colstr = 'slope  ridge_intensity'
    cols = colstr.split()
    df = df[cols]
    df.loc[:, 'slope'] = df.loc[:, 'slope'].abs().values
    x, y = df.loc[:, 'slope'], df.loc[:, 'ridge_intensity']

    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'o', label='original '+colstr)
    ax.plot(x, intercept + slope*x, 'r', label='ridge_intensity={:.2f}slope+{:.2f}'.format(slope, intercept))
    ax.set_xlim([0, x.max()*1.1])
    ax.set_ylabel(r"ridge_intensity")
    ax.set_xlabel("slope")
    plt.legend()
    plt.show()
plot_reg2(ca.df)


# In[ ]:


CWD


# In[ ]:


import os
from subprocess import call
datapath  = r'D:\code\Python\apps\MXP\ContourSelect\samplejob\h\data\dummydb\MXP\job1'
resultpath =r'D:\code\Python\apps\MXP\ContourSelect\samplejob\h\cache\dummydb\result\MXP\job1'
for item in os.listdir(datapath):
    try:
        os.symlink(os.path.join(datapath, item), os.path.join(resultpath, item))
    except OSError:
        call('ln -s {} {}'.format(os.path.join(datapath, item), 
            os.path.join(resultpath, item)), shell=True)


# In[ ]:


import re
re.sub(r'./', '/', './test./key')[1:]
sys.version


# In[ ]:


'test@2/key@1/value@1'.split('/')


# In[ ]:


print('test'.split('@'))
print('test@2'.split('@'))
print(['0'].append(1))


# Data Structure
# 
# A parent Node: (key, [])
# A Leave Node: (key, value)
# 
# Example:
# 
# 1. 
# ('test/value': 213.0, 'test/value@1': 212.0, 'test@1/value': 211.0, 'test@2/value': 210.0)
# [(test, [(value, 213), (value, 212)]), (test, [(value, 211)], (test, [(value, 210)])
#  
# 2. 
# test/options/enable   test/value  test/value@1      test@2/key/option  test@2/value  
# 1-2000     213.0         212.0  revive_bug=Ticket111         210.0
# 
# 
# Paths-indice, value
# ([(test, 0), (options, 0), (enable, 0)], 1-2000)
# ([(test, 0), (value, 0) ], 213.0)
# ([(test, 0), (value, 1) ], 212.0)
# ([(test, 2), (key, 0), (option, 0) ], revive_bug=Ticket111)
# ([(test, 2), (value, 0)], 210)
# 
# 
# (test, [(options, [(enable, )])])
# 
# 
#  
#  
