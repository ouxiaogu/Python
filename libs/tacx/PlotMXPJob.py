"""
-*- coding: utf-8 -*-
Created: peyang, 2018-01-30 14:24:25

Last Modified by:  ouxiaogu

MXPJobPlot: some frequently used plot for MXP job
"""

from MxpJob import MxpJob
import matplotlib.pyplot as plt
import os.path
import pandas as pd
import numpy as np
import cv2
import itertools

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
from logger import logger
log = logger.getLogger(__name__)

#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../imutil")
#from ImDescriptors import calcHistSeries, hist_rect

fig = plt.figure()
AX = fig.add_subplot(111)

# JOBPATHS = [r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/Calaveras_v2/peyang/jobs/9CalavarasV3/DLSEMDataPrepare_yunbo', 
#             r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/Calaveras_v2/peyang/jobs/9CalavarasV3/DLSEMDataPrepare']

# JOBPATHS = [r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/Calaveras_v2/peyang/jobs/9CalavarasV3/DLSEMDataPrepare_yunbo/h/cache/dummydb/result/MXP/job1/DLSEMModelDataPrepare570result1/1',
#             r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/Calaveras_v2/peyang/jobs/9CalavarasV3/DLSEMDataPrepare/h/cache/dummydb/result/MXP/job1/DLSEMModelDataPrepare570result1/1']

JOBPATHS = ['/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/IMEC/Case02_calaveras_v3/3Tmp/ContourSelection/003_D2DB_AIO_from_400_contourSel_452_453']
ENABLES = [500, 503]

from FileUtil import gpfs2WinPath
JOBPATHS = list(map(gpfs2WinPath, JOBPATHS))

class MXPStageData(object):
    """docstring for MXPStageData"""
    def __init__(self, jobpaths, stage=None, stagename=None, enables=None, result_option=None):
        self.jobpaths = jobpaths if isinstance(jobpaths, list) else [jobpaths]
        self.stage = stage
        self.stagename = stagename
        self.enables = enables
        self.result_option = result_option
        self.__readDataSet()

    def __readDataSet(self):
        '''only parse costwt>0 patterns, and use pattern name as index'''
        dataset = []
        for jobpath, enable in itertools.product(self.jobpaths, self.enables):
            job = MxpJob(jobpath)
            df = job.getStageResultFactory(stage=self.stage, stagename=self.stagename, enable=enable, 
                result_option=self.result_option)
            #df = df.loc[df.costwt>0, :].set_index('name')
            dataset.append(df)
        self.dataset = dataset

    def getTwoStageDiff(self, columns=None):
        dataxs = self.dataset
        if columns is not None:
            dataxs = [ df.loc[:, columns] for df in self.dataset] 
        else:
            columns = dataxs[0].columns
        print(dataxs[0].columns)
        merged = dataxs[0].merge(dataxs[1], how='inner', left_index=True, right_index=True, sort=True, suffixes=('_0', '_1'))
        for colname in columns:
            try:
                merged.loc[:, colname+'_diff'] = merged.loc[:, colname+'_0'].sub(merged.loc[:, colname+'_1'])
            except:
                pass
        return merged


def plot_scatter_xy():
    columns = ['offset_x', 'offset_y']
    stagedata = MXPStageData(JOBPATHS, enables=ENABLES)
    diffdata = stagedata.getTwoStageDiff(columns=columns)
    diffcols = [col+'_diff' for col in columns]
    print(diffdata.describe())
    diffdata.plot.scatter(x=diffcols[0], y=diffcols[1])

def plot_pattern_rms():
    columns = ['pattern_rms']
    stagedata = MXPStageData(JOBPATHS, enable=ENABLES)
    diffdata = stagedata.getTwoStageDiff(columns=columns)
    print(diffdata.describe())
    x = np.arange(1, len(diffdata.index)+1)
    for suffix in ['_0', '_1', '_diff']:
        AX.plot(x, diffdata.loc[:, columns[0]+suffix].values, linestyle='--', linewidth=1, marker= 'o', markersize=2.5, label= columns[0]+suffix)
    AX.set_xticks(x) # x[0:-1:100]
    AX.set_xticklabels(diffdata.index, rotation=270)
    addLegend([AX])

def plot_src_tgt_images():
    from ImGUI import imshowMultiple_TitleMatrix, read_pgm
    job_yb, job_oy = JOBPATHS
    print("job_yb", job_yb)
    src1 = read_pgm(job_yb+'/simulated_sem_image.pgm')
    src2 = read_pgm(job_oy+'/simulated_sem_image.pgm')
    dst1 = read_pgm(job_yb+'/real_sem_image.pgm')
    dst2 = read_pgm(job_oy+'/real_sem_image.pgm')

    imshowMultiple_TitleMatrix([src1, dst1, src2, dst2], 4, 1, row_titles=['simulated_sem_image_0', 'real_sem_image_0', 'simulated_sem_image_1', 'real_sem_image_1'], col_titles=['pattern 1'])

def match_src_tgt_images():
    from ImGUI import imshowMultiple_TitleMatrix, read_pgm

    job_yb, job_oy = JOBPATHS
    src1 = read_pgm(job_yb+'/simulated_sem_image.pgm')
    src2 = read_pgm(job_oy+'/simulated_sem_image.pgm')
    dst1 = read_pgm(job_yb+'/real_sem_image.pgm')
    dst2 = read_pgm(job_oy+'/real_sem_image.pgm')
    
    nx, ny = 400, 400
    match_method = cv2.TM_CCORR_NORMED

    templ = dst1[0:nx, 0:ny]

    templ = (templ/65535.).astype(np.float32)
    fixed = (dst2/65535.).astype(np.float32)
    result = cv2.matchTemplate(fixed, templ, match_method)
    cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
    _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
    print('maximum corr:', _maxVal, maxLoc)
    print('maximum corr:', _minVal, minLoc)

    offset_x, offset_y = maxLoc
    im_y, im_x = dst1.shape
    dst1_aligned = np.zeros_like(dst1)
    dst1_aligned[:, offset_x:] = dst1[:, :(im_x-offset_x)]
    src1_aligned = np.zeros_like(src1)
    src1_aligned[:, offset_x:] = src1[:, :(im_x-offset_x)]

    def alignedCropAndHist(im):
        imslice = im[:, offset_x:] # cropped aligned images into max common
        imnorm = np.round(imslice/256).astype(np.uint8)
        return imslice, hist_rect(imnorm)
    imgs = [src1_aligned, dst1_aligned, src2, dst2]
    plots = []
    for rawim in imgs:
        im, hist = alignedCropAndHist(rawim)
        plots.extend([im, hist])

    imshowMultiple_TitleMatrix(plots, 4, 2, 
        row_titles=['simulated_sem_image_0_offset{}'.format(offset_x), 'real_sem_image_0_offset{}'.format(offset_x), 
        'simulated_sem_image_1', 'real_sem_image_1'], 
        col_titles=['pattern 1', 'aligned&cropped ROI histo'], cbar=True)

def barplot(df):
    df.plot(ax=AX, kind='bar')
    AX.set_xticklabels(df.index, rotation=90)

def histplot_rstChkResults(jobs):
    STGNAME = "DLSEMModelDataPrepare"
    COLNAME = "offset_x"
    OPTION = "occfs"
    if isinstance(jobs, MxpJob):
        jobs = [jobs]
    hists = pd.DataFrame()
    lsuffix_ = '_'+os.path.basename(os.path.normpath(jobs[0].jobpath))
    for ix, job_ in enumerate(jobs):
        rsuffix_ = '_'+os.path.basename(os.path.normpath(job_.jobpath))
        dfs_ = []
        stages_ = job_.getAllMxpStages()
        m_stages = [ ''.join(map(str, stg)) for stg in stages_ if stg[0] == STGNAME]
        if len(m_stages)==0:
            raise ValueError("Can't find %s stage in job %s" % (STGNAME, job_.jobpath))
        for stg in m_stages:
            dfs_.append(job_.getStageResultFactory(stg, result_option=OPTION))
        print(dfs_[-1].columns)
        histDF = calcHistSeries(dfs_, column=COLNAME)
        histDF.columns = m_stages
        if(ix == 0):
            hists = histDF
        else:
            hists = hists.join(histDF, lsuffix=lsuffix_, rsuffix=rsuffix_)
            lsuffix_ = ""
    barplot(hists)
    AX.set_title("{} Result comparison".format(STGNAME))
    AX.set_xlabel("{} Ranges".format(COLNAME))
    AX.set_ylabel("Counts")
    plt.show()


def plot_scatter_xys():
    columns = ['offset_x', 'offset_y']
    enables = [500, 502, 503]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, enable in enumerate(enables):
        stagedata = MXPStageData(JOBPATHS, enable=enable)
        dataset = stagedata.dataset[0].loc[:, columns].astype(np.float32)
        #print(dataset.columns, dataset.shape, dataset.head(2), sep='\n')
        #raise ValueError('-')
        dataset.plot.scatter(x='offset_x', y='offset_y', c=COLORS[i], label='D2DB '+str(enable), ax=ax, )
    plt.show()


def compareHist():
    #df.loc[:, 'slope'] = df.loc[:, 'slope'].abs()
    stagedata = MXPStageData(JOBPATHS, enables=ENABLES, result_option='osumccfs')
    dataset = stagedata.dataset
    allColNames = ['similarity', 'center_shift', 'final_cost']
    print(dataset[0].columns)
    for num, alpha in enumerate(allColNames):
        plt.subplot(1, 3, num+1)

        Type0, Type1 = dataset[0][alpha], dataset[1][alpha]
        print(Type0.describe(), Type1.describe(), sep='\n')
        plt.hist((Type0, Type1), bins=25, alpha=0.5, color=['b', 'g'], label=['Normal D2DB 500', 'Rule Model D2DB 503'])
        plt.legend(loc='upper right')
        plt.title(alpha)
        plt.yscale('log')
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

if __name__ == '__main__':
    #plot_scatter_xy()

    # plot_pattern_rms()

    # plot_src_tgt_images()

    # match_src_tgt_images()

    #plot_scatter_xys()
    
    compareHist()