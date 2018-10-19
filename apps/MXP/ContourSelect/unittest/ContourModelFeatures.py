
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

caldatafile = r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/IMEC/Case02_calaveras_v3/3Tmp/CT_KPI_test/Calaveras_v3_regular_CT_KPI_003_slope_modified_revert_all_patterns/h/cache/dummydb/result/MXP/job1/ContourSelectModelCalibration430result1\caldata.txt'
caldatafile = gpfs2WinPath(caldatafile)

df = pd.read_csv(caldatafile, sep='\s+')

tgtColName = 'UserLabel'
neighborColNames = ['NeighborOrientation', 'NeighborParalism']#, 'NeighborParalism', 'NeighborContinuity',


# In[2]:


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


# In[ ]:


def sephist(col):
    TP = df[df[tgtColName] == 0][col]
    TN = df[df[tgtColName] == 1][col]
    return TP, TN
labels = ['intensity', 'slope', 'ridge_intensity', 'contrast', 'EigenRatio']
for num, alpha in enumerate(neighborColNames+labels):
    plt.subplot(2, 4, num+1)

    TP, TN = sephist(alpha)
    plt.hist((TP, TN), bins=25, alpha=0.5, label=map(''.join, zip(2*[tgtColName], 2*['=='], ['0', '1'])), color=['b', 'g'])
    #plt.hist(TP, bins=50, alpha=0.5, label=tgtColName+'==0', color='b')
    #plt.hist(TN, bins=50, alpha=0.5, label=tgtColName+'==1', color='g')
    plt.legend(loc='upper right')
    plt.title(alpha)
    plt.yscale('log')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# In[ ]:


df['NeighborParalism'].describe()


# In[ ]:


df.loc[df[tgtColName]==1, neighborColNames].describe()


# In[ ]:


df.loc[df[tgtColName]==0, neighborColNames].describe()

