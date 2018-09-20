# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-19 15:38:14

SEMContour class test and plots

Last Modified by:  ouxiaogu
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import sys
import os.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
from SEMContour import *
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../../common")
from PlotConfig import *
from FileUtil import gpfs2WinPath

CWD = r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/IMEC/Case02_calaveras_v3/3Tmp/CT_KPI_test/Calaveras_v3_regular_CT_KPI_003_slope_modified_copy/h/cache/dummydb/result/MXP/job1/ContourExtraction400result1'
CWD = gpfs2WinPath(CWD)

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
        self.df = contour.cvtToDf()

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

def plot_reg(df):
    colstr = 'slope  ridge_intensity'
    cols = colstr.split()
    #df = df[cols]
    df.loc[:, 'slope'] = df.loc[:, 'slope'].abs().values
    x, y = df.loc[:, 'slope'], df.loc[:, 'ridge_intensity']

    # from scipy import stats
    # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    #plt.plot(x, intercept + slope*x, 'r', label='fitted ridge_intensity')
    import statsmodels.api as sm
    X = sm.add_constant(x, prepend=False)
    results = sm.OLS(y, X).fit()
    print(results.summary())
    print(results.mse_resid, results.mse_total)
    print(results.params, type(results.params))
    k, b = results.params.loc['slope'], results.params.loc['const']
    
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
    pred_std, predict_ci_low, predict_ci_upp = wls_prediction_std(results)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'o', label='original ridge_intensity v.s. slope')
    y_pred = results.predict()
    ax.plot(x, y_pred, 'r', label='predicted ridge_intensity={:.3f}slope+{:.3f}, $R^2={:.3f}$'.format(k, b, results.rsquared))
    plt.plot(x, predict_ci_low, 'b--', lw=1, label='predict lower')
    plt.plot(x, predict_ci_upp, 'g--', lw=1, label='predict upper')
    
    df.loc[:, 'predict_ci_low'] = predict_ci_low
    df.loc[:, 'predict_ci_upp'] = predict_ci_upp
    
    #ax.set_xlim([0, x.max()*1.1])
    plt.legend()
    plt.show()
    
    flt = (df.ridge_intensity>=df.predict_ci_low)& (df.ridge_intensity<=df.predict_ci_upp)
    
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    print(df.columns)
    ax.plot(df.loc[flt ,'offsetx'], 1024-1-df.loc[flt, 'offsety'], 'b.', markersize=3, label='ridge_intensity In prediction range')
    ax.plot(df.loc[~flt ,'offsetx'], 1024-1-df.loc[~flt, 'offsety'], 'r.', markersize=3, label='ridge_intensity Out prediction range')
    plt.legend()
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
    
    print(ols_quantile(results, X, 0.5))
    
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

if __name__ == '__main__':
    # plot_corr()

    plot_reg()