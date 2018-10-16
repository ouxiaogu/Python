"""
-*- coding: utf-8 -*-
Created: peyang, 2018-04-20 2:33:40 PM

FemplusJob: Class to hold tachyon FEM+ job, derived from class TachyonJob
"""

from TachyonJob import Job
from collections import OrderedDict
import os.path
import re
from XmlUtil import getConfigData

# TACX GUI classes
import sys
sys.path.insert(0, r'/n/filer3b/home/dev/qsun/gui_10/build_root/libs')
import FEMPlusJob
import jobresult
import mod
import Tac
import FEMSetup
import FEMReview
import ControlCenter

logger.initlogging(debug=False)
import logger
log = logger.getLogger(__name__)

"""global keys"""
nameconfig = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'filekey_GUICol.txt')
filekeyGUICol = pd.read_csv(nameconfig,sep='\t',index_col=0)
defaultGUICol = ['AI CD','ADI Model CD','Model CD','Model Error','ILS Result (1/um)']
import string
illChar=string.punctuation+string.whitespace

class FemplusJob(Job):
    """docstring for FemplusJob"""
    def __init__(self, jobpath):
        super(FemplusJob, self).__init__(jobpath)
        self.__openJob()

    def __enter__(self, jobpath):
        self.__openJob()
        return self

    def __exit__(self, exc_type, exc_value, traceback):

    def getFlow(self):
        pass

    def isCalJob(self):
        pass

    def __openJob(self):
        ''' The entrance function'''
        self.

    def __getReleaseDir(self):


    def __getRelResultPath(self):
        """Get relative result path to jobroot where the *result.txt, femresult.xml, process.xml located"""
        xml=os.path.join(self.__jobPath,'jobinfo','job_result.xml')
        source=ET.parse(xml)
        for p in source.getiterator('value'):
            path = os.path.split(p.text)[0]
        path = path[1:] if path[0] == '/' else path
        return path

    def getGaugeFilePath(self, condid):
        """get the gauge file path of a given condid"""
        t = self.getConditions()
        return os.path.join(self.__jobPath,t['local'])

    def getConditions(self):
        '''
        execute sql to get all conditon id,FEM condtion,and used gauge file path, etc
        '''
        result = query(r'select * from femcondition;', jobpath, PerlDir)
        result = pd.read_csv(StringIO(result), sep='\t')
        idx = result['local'].isnull()
        result=result.astype(object)
        result.ix[idx,'local']=result.ix[idx,'source']

        for var in ['defocus', 'dose']:
            result[var] = str_extract(result.settings, r'{}=(?P<{}>[\d.-]+)'.format(var, var))

        return result

    def getGaugeFile(self, conditionID):
        """get the gaugefile into a dataframe of a given conditionID"""
        filepath=self.__getGaugeFilePath(conditionID)
        log.info('reading gauge file: '+filepath+' conditionID '+str(conditionID))
        try:
            df=pd.read_csv(filepath,sep=' ',index_col=0)
        except:
            with open(filepath,'r') as fid:
                lines=fid.readlines()
            newline='\n'.join([x.strip() for x in lines])
            df=pd.read_csv(StringIO(newline),sep=' ',index_col=0)
        if len(df.columns)<7:#gauge seperator not right
            df=pd.read_csv(filepath,sep='\t',index_col=0)
        return df

if __name__ == '__main__':
    jobpath = r'/gpfs/DEV/FEM/peyang/release/E8.0/MOD9944/job2_2D_CD+EP_align_correction_GN'
    from FileUtil import gpfs2WinPath
    jobpath = gpfs2WinPath(jobpath)
    with FemplusJob(jobpath) as m_job:
        print m_job.getConditions()