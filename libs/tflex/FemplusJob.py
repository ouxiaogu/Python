# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-07-12 10:24:09

FemplusJob: Class to hold tachyon FEM+ job, derived from class TachyonJob

Last Modified by:  ouxiaogu
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../tacx")
from TachyonJob import Job

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
from XmlUtil import getConfigData
from logger import logger
log = logger.getLogger(__name__)

class FemplusJob(Job):
    filekey_GUICol_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'filekey_GUICol.txt')
    filekey_GUICol = pd.read_csv(filekey_GUICol_path, sep='\t', index_col=0)
    omit_GUICols = ['Error %', 'Model CD', 'Through Cond. Error']

    def __init__(self, jobpath):
        super(FemplusJob, self).__init__(jobpath)
        self.__openJob()

    def __enter__(self, jobpath):
        self.__openJob()

    def __openJob(self):
        ''' The entrance function'''
        pass

    def __getRelResultPath(self):
        """Get relative result path to jobroot where the *result.txt, femresult.xml, process.xml located"""
        xml=os.path.join(self.__jobPath,'jobinfo','job_result.xml')
        source=ET.parse(xml)
        for p in source.getiterator('value'):
            path = os.path.split(p.text)[0]
        path = path[1:] if path[0] == '/' else path
        return path

    @staticmethod
    def rms(error_, wt_):
        flt = np.where(wt_ > 0)
        error = error_[flt]
        wt = wt_[flt]
        if len(error) == 0:
            return np.nan
        else:
            return np.sqrt(np.sum((np.power(error, 2) * wt))/np.sum(wt))

if __name__ == '__main__':
    jobpath = r'/gpfs/DEV/FEM/peyang/release/E8.0/MOD9944/job2_2D_CD+EP_align_correction_GN'
    from FileUtil import gpfs2WinPath
    jobpath = gpfs2WinPath(jobpath)
    with FemplusJob(jobpath) as m_job:
        print m_job.getConditions()