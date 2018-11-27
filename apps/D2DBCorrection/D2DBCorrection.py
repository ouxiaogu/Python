# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-11-22 10:47:20

FEM+ D2DB residual error correction task

Last Modified by:  ouxiaogu
"""

import glob
import re
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../../libs/tacx")
from femplusAPI import openJob, isCheckJob, FEMJob, TacJob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../../libs/common")
from FileUtil import splitFileName
from logger import logger
log = logger.getLogger(__name__)

class D2DBCorrection(object):
    LUAFILE = os.path.dirname(os.path.abspath(__file__))+"/../../../../lua/customized/PatternAlignCorrection.lua"
    ERR_FILE_PREFIX = 'cluster_shiftd_error_result'
    Used_Spec_Cols = ['SEM', 'CENTER_X', 'CENTER_Y']
    Used_Gauge_Cols = ['GaugeClusterId', 'cluster_shift_x', 'cluster_shift_y']

    def __init__(self, jobpath, semspec='', outspec='', reuse_result=True, reuse_lua=False, **kwargs):
        self.job = openJob(jobpath)
        self.jobpath = jobpath
        self.semspec = semspec
        self.outspec = outspec
        self.reuse_result = reuse_result
        self.reuse_lua = reuse_lua
        log.debug("{}".format(self.__dict__))

        self.__build()

    def __build(self):
        self.__validate()        

        if not os.path.exists(self.semspec):
            log.warning("SEM Spec file doesn't exist at: {}!".format(self.semspec))
            self.wiSpec = False
            return
        try:
            semspectable = pd.read_csv(self.semspec, sep='\t')
            if any([col not in semspectable.columns for col in D2DBCorrection.Used_Spec_Cols]):
                log.error("Can't correctly parse {} from {}!".format(D2DBCorrection.Used_Spec_Cols, self.semspec))
                return
        except:
            log.error("Can't correctly parse SEM Spec file {}!".format(self.semspec))
            return
        self.wiSpec = True
        self.semspectable = semspectable

        outnames = None
        if not os.path.exists(os.path.dirname(self.outspec)):
            log.warning("Output SEM Spec folder doesn't exist at: {}, use job path instead!".format(os.path.dirname(self.outspec)))
        else:
            if os.path.isfile(self.outspec):
                outnames = splitFileName(self.outspec)
            elif os.path.isdir(self.outspec):
                outnames = [self.outspec, 'newspec', 'txt']
        outnames = [self.jobpath, 'newspec', 'txt'] if outnames is None else outnames
        self.outnames = outnames

    def __validate(self):
        if not isCheckJob(self.jobpath):
            raise TypeError('Job type is not "FEM+ check" for {} !'.format(self.jobpath))

    def reusable(self):
        if self.reuse_result:
            if self.job.status in FEMJob.FINISHSTATUS:
                errpathex = os.path.join(self.job.resultabspath, D2DBCorrection.ERR_FILE_PREFIX+'*')
                errfiles = glob.glob(errpathex)
                if len(errfiles) > 0:
                    log.info("Pattern align correction results is reusable at: {}".format(self.jobpath))
                    return True
                else:
                    log.warning("Job result is not reusable since job doesn't contain pattern align correction result columns")
            else:
                log.warning("Job result is not reusable since job {}".format(self.job.status))
        return False

    def submitjob(self):
        if not self.reuse_lua:
            with open(D2DBCorrection.LUAFILE) as fh:
                joblua = fh.read()
            try:
                self.job.setlua('job', joblua)
                self.job.save()
            except AttributeError:
                log.debug("tacx fem+ api,  'CheckJob' object has no attribute 'setlua' yet")
                jobluafile = os.path.join(self.job.dataabspath, 'lua', 'job1.lua')
                with open(jobluafile, 'w') as fo:
                    fo.write(joblua)
                    log.info("Successfully set pattern align correction lua into {}".format(jobluafile))
        else:
            log.info("Reuse the pattern align correction lua in the job")

        ''' # tac job submit not work here
        tacjob = TacJob(self.jobpath)
        tacjob.submit()
        '''
        self.job.submit()

        log.info("job is successfully submitted, please wait job done ...")

    def updateProcesses(self):
        processtable = self.job.result.getProcessTable()
        filekey_GUICol = pd.read_csv(FEMJob.filekey_GUICol_path, sep='\t')
        extra_cols = ['cost_wt', 'GaugeClusterId']

        minRms = (float("inf"), '')
        for _, row in processtable.iterrows():
            processid = row.loc['Process']
            condtable = self.job.result.conditiontable(processid)
            resultSet = self.job.result.gaugesetWResCondId(processid, condtable, True, extra_cols)
            
            oriErrKey = 'Model Error'
            newErrKey = 'cluster_shiftd_error' if 'cluster_shiftd_error' in resultSet.columns else 'cluster_shifted_error'
            if newErrKey not in resultSet.columns:
                log.error("Cannot find key {} in job results table! Please check job lua setting, increase rmsUpperBound & maxShiftRange if necessary!".format(newErrKey))
                sys.exit(-1)
            newErrFunc = lambda s: s[oriErrKey] if np.isnan(s[newErrKey]) or s[newErrKey] == 0 else s[newErrKey]
            resultSet.loc[:, newErrKey] = resultSet.apply(newErrFunc, axis=1)
            oriRms = FEMJob.rms(resultSet.loc[:, oriErrKey].values, resultSet.loc[:, 'cost_wt'].values)
            newRms = FEMJob.rms(resultSet.loc[:, newErrKey].values, resultSet.loc[:, 'cost_wt'].values)
            
            descr = "original RMS {:.4f} cluster shifted RMS {:.4f}".format(oriRms, newRms)
            log.info("Process {}, {}".format(processid, descr))
            if newRms < minRms[0]:
                minRms = (newRms, descr)
            descr = re.sub('\s', '_', descr)

            params = row.to_dict()
            # log.debug("process {}, params: {}".format(processid, params))
            params["(cluster_shifted_rms)"] = str(newRms)
            for key in FEMJob.omit_process_Cols:
                try:
                    params.pop(key, None)
                except:
                    pass
            # log.debug("process {}, params updated: {}".format(processid, params))

            results = {}
            for col in resultSet.columns:
                if col not in extra_cols + FEMJob.omit_gauge_Cols:
                    try:
                        filekey = filekey_GUICol.loc[filekey_GUICol['GUI Col']==col, 'filekey'].values[0]
                    except:
                        filekey = col+"_result"
                    results[filekey] = resultSet.loc[:, col].unstack()
            msg = self.job.result.updateProcess(processid, descr, params, results)
            log.info(msg)

            if self.wiSpec:
                self.updateSpec(processid, resultSet)
        self.job.setJobInfoConfigData('descr', minRms[1])

    def updateSpec(self, processid, resultSet):
        corrections = resultSet[D2DBCorrection.Used_Gauge_Cols]
        flt = np.logical_not(
                np.logical_or(
                    np.logical_or(
                        pd.isna(corrections['cluster_shift_x']),
                        pd.isna(corrections['cluster_shift_y'])),
                    np.logical_and(corrections['cluster_shift_x'] == 0,
                                   corrections['cluster_shift_y'] == 0)))

        clusterinfo = corrections.loc[flt, :].groupby("GaugeClusterId").first()
        outspectable = self.semspectable.copy(deep=True)
        log.debug("cluster info: \n{}".format(clusterinfo))
        for name, cluster in clusterinfo.iterrows():
            try: 
                outspectable.loc[outspectable['SEM']==name, 'CENTER_X'] += cluster.loc['cluster_shift_x']
                outspectable.loc[outspectable['SEM']==name, 'CENTER_Y'] += cluster.loc['cluster_shift_y']
            except:
                log.warning("GaugeClusterId {} cannot be found in input SEM spec file: {}!".format(name, self.semspec))
        outspecfile = os.path.join(self.outnames[0], '{}_{}.{}'.format(self.outnames[1], processid, self.outnames[2]))
        outspectable.to_csv(outspecfile, index=False, sep='\t')
        log.info("Successfully update SEM spec file for process {} into {}".format(processid, outspecfile))

    def run(self):
        if not self.reusable():
            self.submitjob()
            try:
                self.job.waitTillDone()
            except RuntimeError:
                sys.exit("Job {}! Please check job terminate reason, change the job setting and rerun the FEM+ D2DB correction external tool again! ".format(self.job.status))

        self.updateProcesses()