"""
-*- coding: utf-8 -*-
Created: peyang, 2018-01-30 14:24:25

Last Modified by: peyang

MXPJobPlot: some frequently used plot for MXP job
"""

from MXPJob import MXPJob
import PlotDataPrep as pdp
from PlotConfig import *
import matplotlib.pyplot as plt
import logger
import os.path
import pandas as pd

logger.initlogging(debug=False)
log = logger.getLogger("MXPJobPlot")

def barplot(df):
    df.plot(ax=AX, kind='bar')
    AX.set_xticklabels(df.index, rotation=90)

def histplot_rstChkResults(jobs):
    STGNAME = "ResistModelCheck"
    COLNAME = "pattern_rms"
    OPTION = "osumkpis"
    if isinstance(jobs, MXPJob):
        jobs = [jobs]
    hists = pd.DataFrame()
    lsuffix_ = os.path.basename(os.path.normpath(jobs[0].jobpath))
    for ix, job_ in enumerate(jobs):
        rsuffix_ = os.path.basename(os.path.normpath(job_.jobpath))
        dfs_ = []
        stages_ = job_.getAllMxpStages()
        m_stages = [ ''.join(map(str, stg)) for stg in stages_ if stg[0] == STGNAME]
        if len(m_stages)==0:
            raise ValueError("Can't find %s stage in job %s" % (STGNAME, job_.jobpath))
        for stg in m_stages:
            dfs_.append(job_.getStageResultFactory(stg, option=OPTION))
        histDF = pdp.calcHist(dfs_, column=COLNAME)
        histDF.columns = m_stages
        if(ix == 0):
            hists = histDF
        else:
            hists = hists.join(histDF, lsuffix='_'+lsuffix_, rsuffix='_'+rsuffix_)
    barplot(hists)
    AX.set_title("{} Result comparison".format(STGNAME))
    AX.set_xlabel("{} Ranges".format(COLNAME))
    AX.set_ylabel("Counts")
    plt.show()

if __name__ == '__main__':
    jobpaths = [r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/Calaveras_v2/peyang/jobs/8GF02/Case3E_GF_EP5_study_c2c_id2db_v1', r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/Calaveras_v2/peyang/jobs/8GF02/Case3E_GF_EP5_study_c2c_id2db_v2_BSEonly', r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/Calaveras_v2/peyang/jobs/8GF02/Case3E_GF_EP5_study_c2c_id2db_v3_old']
    from FileUtil import gpfs2WinPath
    jobpaths = map(gpfs2WinPath, jobpaths)
    jobs=[]
    for jobpath in jobpaths:
        job = MXPJob(jobpath)
        jobs.append(job)
    histplot_rstChkResults(jobs)
    #plt.close(FIG)