"""
-*- coding: utf-8 -*-
Created: peyang, 2018-01-24 21:06:48

Last Modified by: peyang

Gauge: Gauge Utility module and Gauge Class(will add when necessary)
"""

import pandas as pd
import numpy as np
import math

import logger
logger.initlogging(debug=False)
log = logger.getLogger("Gauge")

def stdCol(name, illchars = string.punctuation+string.whitespace):
    newname = name.lower()
    for char in illchars:
        newname = newname.replace(char, '_')
    if newname=='drawn_cd':
        newname =  'draw_cd'
    if "ils" in newname:
        newname = 'ils'
    return newname

def stdDFCols(df0):
    df = df0.copy()
    for name in df.columns.values:
        df.rename(columns={name: stdCol(name)}, inplace=True)
    return df

def stdGauge(df0, tostdcol=False):
    df = df0.copy()
    if tostdcol:
        df = stdDFCols(df)
    df = df[df["cost_wt"] > 1e-6 ]
    # df = df[df['model_cd'] > 1e-6]
    df = df[df['wafer_cd'] > 1e-6]
    # df = df[df['ai_cd'] > 1e-6]
    df.index = range(len(df))
    return df

def validTitle(title, illChars=string.punctuation+string.whitespace):
    newTitle = title.lower()
    newTitle = newTitle.replace('<', '_le_')
    newTitle = newTitle.replace('>', '_gt_')
    for x in illChars:
        newTitle=newTitle.replace(x,'_')
    return newTitle

def mergeColumns(df0, columns, newcolname,**args):
    '''merge listed columns into a new column in the dataframe, another added column is category'''
    df = df0.copy(deep=True)
    df['category'] = ''
    firstInstance = True
    for colname in columns:
        df['category'] = colname
        df[newcolname] = df[colname]
        if firstInstance:
            df_result = df.copy(deep=True)
            firstInstance = False
        else:
            df_result = df_result.append(df)
    log.debug("merged DF:")
    log.debug("%s" % str(df_result.pivot_table(values='gauge', columns='category', aggfunc=np.count_nonzero )))
    return df_result

def groupsort(df, mode=''):
    '''sort DataFrame group by group, uring pre-defined mode:
            'cost_wt': largest cost_wt come first
    '''
    if len(df)==0:
        return df
    grouplabel = [x for x in df.columns if 'group' in x.lower()][0]
    grouped = df.groupby([grouplabel])
    if mode=='count':
        pass
    else: # default mode, 'cost-wt'
        wt_label = [x for x in df.columns if 'wt' in x.lower()][0]
        sumwt = grouped.apply(lambda x: x[wt_label].sum())
        sumwt = sumwt.sort_values(ascending=False)
        sortedgroups = sumwt
    first = True
    result = pd.DataFrame({})
    for group in sortedgroups.index:
        curdf = df[df[grouplabel]==group]
        if first:
            result = curdf
            first = False
        else:
            result = result.append(curdf)
    return result

def errDistributionPlotSort(df):
    grouplabel = [x for x in df.columns if 'group' in x.lower()][0]
    drawcd_label = [x for x in df.columns if 'draw' in x.lower()][0]
    gauge_label = [x for x in df.columns if 'gauge' in x.lower()][0]
    type_label = [x for x in df.columns if 'type' in x.lower()][0]
    gauge1D = df[df[type_label]=='1D']
    gauge2D = df[df[type_label]=='2D']
    gauge1D = gauge1D.sort_values(by=[drawcd_label, gauge_label])
    gauge1D = groupsort(gauge1D)
    gauge2D = gauge2D.sort_values(by=[drawcd_label, gauge_label])
    gauge2D = groupsort(gauge2D)
    result = gauge1D.append(gauge2D)
    groups = result[grouplabel].unique()
    return result, groups

def calRMS(df0, option=''):
    df = stdGauge(df)
    wt_label = [x for x in df.columns if 'wt' in x.lower()][0]
    df = df[df[wt_label]>1e-6]
    if len(df)==0:
        return 0
    err_label = [x for x in df.columns if 'error' in x.lower()][0]
    if option=='unweighted':
        df.ix[:, wt_label]=1
    df.ix[:, 'err^2'] = df.apply(lambda x: (x[err_label])**2, axis=1)
    rms = math.sqrt(np.inner(df['err^2'], df[wt_label]) /  df.sum()[wt_label])
    rms = round(rms, 3)
    return rms

def getSpecName(df):
    specnames=[x for x in df.columns if 'spec'==x.lower() or 'range_max'==x.lower()]
    if len(specnames)<1:
        raise KeyError('Fail, neither spec nor range_max column exist')
    else:
        sort_order = {}
        for col in specnames:
            sort_order[col] = 3
            if col.lower() == 'range_max':
                sort_order[col] = 2
            if col.lower() == 'spec':
                sort_order[col] = 1
        specnames = sorted(specnames, key=sort_order.__getitem__)
    return specnames[0]

def countInSpecRatio(df0):
    df = stdGauge(df0)
    errorlabel = [x for x in df.columns if 'error' in x.lower()][0]
    speclabel = getSpecName(df)
    inspecfilter = "({}>-{}) and ({}<{})".format(errorlabel, speclabel, errorlabel, speclabel)
    nGauges = len(df)
    nInSpecGauges = len(df.query(inspecfilter))
    stat_dict = {"InSpecNum": nInSpecGauges,
                 "All": nGauges}
    if nGauges != 0:
        stat_dict["InSpecRatio"] = "{percent:.2%}".format(percent=1.*nInSpecGauges/nGauges)
    else:
        stat_dict["InSpecRatio"] = "{percent:.2%}".format(percent=0.)
    return stat_dict

def calInSpecRatio(df0):
    df=stdGauge(df0)

    df_valid = df.query('cost_wt > 0')
    df_1D = df_valid.query("type=='1D'")
    df_2D = df_valid.query("type=='2D'")
    stat_dict = {}
    stat_dict["1D"] = countInSpecRatio(df_1D)
    stat_dict["2D"] = countInSpecRatio(df_2D)
    stat_dict["All"]= countInSpecRatio(df_valid)
    return pd.DataFrame(stat_dict)