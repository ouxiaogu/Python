# -*- coding: utf-8 -*-
import numpy as np
import os, os.path
import sys
import pandas as pd
sys.path.append('C:\Localdata\D\Note\Python\FEM\util')
import math_util as mathu

workpath = os.path.dirname(os.path.abspath(__file__))

#datafile = os.path.join(workpath, "data", "uncertaintyMSE.txt")
#df = pd.read_csv(datafile, delimiter = '\t')
#def f(x):
#    return x*x
#df["gauge1^2"] = df['gauge1'].apply(lambda x: x*x)
#gauge1_uncert = np.array(df.loc[:,"gauge1^2"].values)
#wt =  np.array(df.loc[:,"wt"].values)
#mse1 = np.sqrt(np.dot(gauge1_uncert, wt)/sum(wt))
#df["gauge2^2"] = df['gauge2'].apply(lambda x: x*x)
#gauge2_uncert = np.array(df.loc[:,"gauge2^2"].values)
#mse2 = np.sqrt(np.dot(gauge2_uncert, wt)/sum(wt))
#df.to_csv(os.path.join(workpath, "data", "uncertaintyMSE2.txt"), sep = '\t', index=False)
#print(mse1, mse2)
#
#
# user function:


#uncertainty= {
#	"Chain_H_1_6": {
#		"defocus": -0.33146004749523,
#		"focusBlur": -0.067607132516969,
#		"InnerCorner": 0.3507597401676,
#		"OuterCorner": -0.77453912324503,
#		"aiBlur": -0.18637674352627,
#		"SWA": -6.1911981680539
#	},
#	"T2S_52_3": {
#		"defocus": 0.23429016103705,
#		"focusBlur": 0.032992448916303,
#		"InnerCorner": -1.1228317108933,
#		"OuterCorner": 4.3794240614883,
#		"aiBlur": 0.75806088291785,
#		"SWA": 13.322553998604
#	},
#	"T2S_53_1": {
#		"defocus": 0.077003613015023,
#		"focusBlur": 0.0029740697800378,
#		"InnerCorner": -0.99741671346712,
#		"OuterCorner": 4.6991302526801,
#		"aiBlur": 0.69435302667105,
#		"SWA": 14.598405748054
#	},
#	"Chain_H_1_12": {
#		"defocus": -0.38963677717297,
#		"focusBlur": -0.082654259636854,
#		"InnerCorner": 0.5457890788044,
#		"OuterCorner": -1.0085334532226,
#		"aiBlur": -0.35348421297209,
#		"SWA": -6.7594550728677
#	},
#	"TruP_12_5": {
#		"defocus": -0.12545543144049,
#		"focusBlur": -0.08957620981451,
#		"InnerCorner": 0,
#		"OuterCorner": 0,
#		"aiBlur": -0.36650411275338,
#		"SWA": -3.7329110241482
#	},
#	"BAR_ISO_1_4": {
#		"defocus": -1.1393315116756,
#		"focusBlur": -0.42681388215797,
#		"InnerCorner": 0.10935729196433,
#		"OuterCorner": -0.2419935264821,
#		"aiBlur": -1.331971386603,
#		"SWA": -3.1982283453667
#	}
#}
#df = pd.DataFrame(uncertainty)
#dict_wt = {'InnerCorner':1, 'OuterCorner':1, 'SWA':1, 'aiBlur': 1,'defocus': 3,'focusBlur': 1}
#length = len(dict_wt)
#wt = range(length)
#ind = [[] for i in range(length)]
#ind = df.index.values.tolist()
#i = 0
#for key in ind:
#    wt[i] = dict_wt[key]
#    i = i + 1
#gauge_name = df.columns.values.tolist()
##print(df)
#df['wt'] = pd.Series(wt, index=ind)
#df.to_csv(os.path.join(workpath, "data", "mse3.txt"), sep = '\t')
#print(df.index.values)
#print(df.columns.values)
#for name in gauge_name:
#    mathu.mse_wt(df, name, "wt", log=True)

##########################
###  read data   #########
##########################
datafile = os.path.join(workpath, "data", "mse4.txt")
df = pd.read_csv(datafile, delimiter = '\t')
df_T = df.set_index(['gauge']).T

gauge_name = df_T.columns.values.tolist()
wt = [1, 1, 1, 1, 3, 1]
#print(gauge_name)
print(df_T.index.values.tolist())
df_T['wt'] = pd.Series(wt, index=df_T.index)
length = len(gauge_name)
mse = range(length)
count = 0
for name in gauge_name:
    mse[count] = mathu.mse_wt(df_T, name, "wt")
    if(count%100 == 0):
        print("{} mse={}".format(name, mse[count]))
    count = count + 1
df['mse'] = pd.Series(mse, index=df.index)    
df.to_csv(os.path.join(workpath, "data", "mse4_result.txt"), sep = '\t', index=False)