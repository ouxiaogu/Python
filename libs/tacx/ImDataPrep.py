"""
-*- coding: utf-8 -*-
Created: peyang, 2018-01-29 09:51:26

Last Modified by: peyang

PlotDataPrep: Plot Data Preparation
"""

import pandas as pd
import numpy as np
import logger
import re

__all__ = [ 'RMS_BIN_RANGES', 'ZNCC_BIN_RANGES',
            'calcHist', 'readDumpImage', 'readBBox',
            ]

RMS_BIN_RANGES = [0, 2, 4, 6, 8, 10, 15, 20, 30, 50, 100]
ZNCC_BIN_RANGES = [0, 0.2, 0.5, 0.8]

logger.initlogging(debug=False)
log = logger.getLogger("PlotDataPrep")

def calcHist(series_, ranges=RMS_BIN_RANGES, column=None):
    """
    calculate the histogram of given Pandas Serires

    Parameters
    ----------
    series_ : Pandas Series to be analyzed
    ranges : Array of the histogram bin boundaries.

    Returns
    -------
    histDF : DataFrame of the hists
    """

    labels = []
    filts = []
    hists = np.zeros((len(series_), len(ranges)))
    nitems = len(ranges)
    log.debug("ranges, %s" % str(ranges))

    for i in range(nitems):
        if i == nitems - 1:
            label = "{}~".format(ranges[i])
            filt  = lambda x, y=ranges[i]: (x >= y) & True
        else:
            label = "{}~{}".format(ranges[i], ranges[i+1])
            filt  = lambda x, y=ranges[i], z=ranges[i+1]: (x >= y) & (x < z)
        labels.append(label)
        filts.append(filt)
    log.debug(labels)
    log.debug("%s" % str([ff==filts[0] for ff in filts]))

    for i, series in enumerate(series_):
        if column is not None:
            try:
                series = series.ix[:, column]
            except KeyError:
                raise KeyError("Input column {} is not in columns {}".format(series.columns))
        if not isinstance(series, pd.Series):
            try:
                series = pd.Series(series)
            except ValueError:
                raise ValueError("Error occurs when converting input into Pandas Series")
        series = series.sort_values()
        log.debug("series_: %s" % str(series.values ))
        for j, filt in enumerate(filts):
            hists[i, j] = len(series.ix[ series.apply(filt) ] )
            log.debug("%s: %d" % (labels[j], hists[i, j]))
    histDF = pd.DataFrame(hists, columns=labels).astype('int32')
    histDF = histDF.transpose() # for barplot, column as range lables
    return histDF

def readDumpImage(infile, skip_header=0):
    im = np.genfromtxt(infile, skip_header=skip_header).astype('float32')
    imat = np.asmatrix(im)
    return imat

def readBBox(infile):
    bbox = None
    with open(infile) as f:
        for i, line in enumerate(f.readlines()):
            if i > 2:
                break
            m = re.search("^BBox: xini = (\d+), xend = (\d+), yini = (\d+), yend = (\d+)", line)
            if m is not None:
                bbox = m.groups()
                break
    return bbox

if __name__ == '__main__':
    np.random.seed(100)
    array_ = np.random.randint(0, 40, 100)
    histDF_ = calcHist([array_])
    print histDF_.describe().ix[:2,:]
