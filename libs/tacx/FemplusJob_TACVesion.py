"""
-*- coding: utf-8 -*-
Created: peyang, 2018-04-20 2:33:40 PM

Refer to Hong-fei's ExpressModel/express_model/common/femplusAPI.py
Last Modified by:  ouxiaogu

FemplusJob: Class to hold tachyon FEM+ job, derived from class TachyonJob

This API is covered on top of FEMPlusJob, which provided basic C++ based API.
By arranging the flatten basic level API into object oriented design, we
provide user an easier and clearer interface to handle FEM job.

FEMPlusJob opens a job, create object in memory.
All get, set methods act on it.
This mechanism can't grantee newest data, because another thread may modify
the job DB or data but the already open object is not aware of it.
To keep working on newest data, we must open the job each time before accessing
data requirement.
"""

from TachyonJob import Job
from collections import OrderedDict
import os.path
import xml.etree.ElementTree as ET
import re

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
from logger import logger
log = logger.getLogger(__name__)

''' There is one class for each flow type
'''
FLOW2CLASS = {
    FEMPlusJob.FLOW_NAME_CAL: (ADICalJob, FEMPlusJob.ModelCalibration),
    FEMPlusJob.FLOW_NAME_GEN: (ADIGenJob, FEMPlusJob.ModelGeneration),
    FEMPlusJob.FLOW_NAME_AUTOCAL: (AutoCalJob, FEMPlusJob.AutoCalibration),
    FEMPlusJob.FLOW_NAME_CHECK: (CheckJob, FEMPlusJob.ModelCheck),
    FEMPlusJob.FLOW_NAME_AEICAL: (AEICalJob, FEMPlusJob.AEIModelCal),
    FEMPlusJob.FLOW_NAME_AEIGEN: (AEIGenJob, FEMPlusJob.AEIModelGen),
    FEMPlusJob.FLOW_NAME_M3D: (M3DLibGenJob, FEMPlusJob.M3dLibGen),
    FEMPlusJob.FLOW_NAME_W3D: (W3DLibGenJob, FEMPlusJob.W3dLibGen),
}

class FemplusJob(Job):
    """docstring for FemplusJob"""
    def __init__(self, jobpath):
        super(FemplusJob, self).__init__(jobpath)
        self.__openJob()

    def __enter__(self, jobpath):
        self.__openJob()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tacjob is not None:
            FEMPlusJob.releaseJob(self.tacjob)

    def getFlow(self):
        basejob = FEMPlusJob.openJob(self.jobpath)
        if basejob is None:
            raise IOError('Open job failed!')
        log.debug('getFemFlowType')
        flowtype = basejob.getFemFlowType()
        FEMPlusJob.releaseJob(basejob)
        return flowtype


    def isCalJob(self):
        flowtype = getFlow(self.jobpath)
        if flowtype == FEMPlusJob.FLOW_NAME_CAL:
            return True
        else:
            return False

    def __openJob(self):
        ''' The entrance function.
        Automatically choose the right class according to the flowtype

        job: instance of a child class of FEMJob
        '''
        flowtype = getFlow(jobpath)
        thisname, apiname = FLOW2CLASS.get(flowtype, FLOW2CLASS.values()[0])
        basejob = apiname.openJob(jobpath)
        if basejob is None:
            raise IOError('Open job failed!')
        self.tacjob = thisname(basejob)

class JobCompoent(object):
    def __init__(self, basejob):
        self._basejob = basejob

class Condition(JobCompoent):
    @property
    def table(self):
        '''
        Returns
        -------
        condition: pandas.DataFrame
            Condition table, containing PW information
        '''
        log.debug('getConditionTable')
        table = self._basejob.getConditionTable()
        temp = [[table.cell(x,y) for y in range(table.columnCount())] for x in range(table.rowCount())]
        return pd.DataFrame(temp, columns=[table.colHeader(x) for x in range(table.columnCount())])

    @property
    def NC(self):
        '''
        Nominal condition definition:
            - delta focus and dose is both 0
            - Not considering multi exposure w/o empty role
        if found such condition, return the index of this condition, not the condition id
        if not found such condition, return -1
        This value will be mainly used in linear solver
        '''
        table = self.table
        ncfilter = (table['Delta_Focus'] == '0') & (table['Delta_Dose'] == '0')
        table = table.ix[ncfilter, :]
        if len(table) == 0:
            return -1
        else:
            return table.index[0]

    def condid_seq(self, condid):
        ids = self.table['ID'].tolist()
        if condid is not None:
            return sorted(ids).index(str(condid))
        else:
            return 0

    def __eq__(self, other):
        tb1 = self.table
        tb2 = other.table
        return np.all(np.array(tb1 == tb2))

class Data(Condition):
    @property
    def gaugeset(self):
        ''' Gauge table containing all the conditions
        '''
        condtable = self.table
        gaugeset = []
        for i, k in enumerate(condtable.index):
            gpath = condtable.ix[k, 'Local Path']
            if gpath == '':
                gpath = condtable.ix[k, 'Updated Gauge Path']
            try:
                gauge = self._readGauge(gpath)
            except IOError:
                log.debug('getPath')
                gpath = os.path.join(self._basejob.getPath(), gpath)
                gauge = self._readGauge(gpath)
            gauge['condid'] = int(i)
            gauge['condwt'] = float(condtable.ix[k, 'Weight'])
            gaugeset.append(gauge)
        columns = gauge.columns
        gauge = pd.concat(gaugeset, axis=0)
        gauge = gauge.reindex(columns=columns)
        gname = gauge.index.name
        return gauge.reset_index().set_index([gname, 'condid'])

    @staticmethod
    def _readGauge(path):
        log.debug('reading gauge file: {}'.format(path))
        with open(path) as f:
            temp = f.readlines()
        temp = '\n'.join([x.strip() for x in temp])
        temp = StringIO(temp.replace('"', QUOTASTR))
        gauge = pd.read_csv(temp, delimiter=r'[\t\s]+').dropna(how='all')
        gaugename = filter(lambda x: re.match(r'^gauge$', x, re.IGNORECASE), gauge.columns)[0]
        cols = gauge.columns.tolist()
        cols[cols.index(gaugename)] = gaugename.lower()
        gauge.columns = cols
        gauge = gauge.set_index(gaugename.lower())
        temp.close()
        return gauge

    def setGaugeset(self, gaugeset):
        condtable = self.table
        gaugeset = gaugeset.reset_index()
        for i, k in enumerate(condtable.index):
            gpath = condtable.ix[k, 'Local Path']
            if gpath == '':
                gpath = condtable.ix[k, 'Updated Gauge Path']
            if not os.path.isfile(gpath):
                log.debug('getPath')
                gpath = os.path.join(self._basejob.getPath(), gpath)
            gauge = gaugeset.ix[gaugeset['condid']==i].drop(['condid'])
            temp = StringIO()
            gauge.to_csv(temp, sep='\t', index=False)
            temp = temp.getvalue().replace(QUOTASTR, '"')
            with open(gpath, 'w') as f:
                f.write(temp)
            log.debug('set gauge for condition {} to {}'.format(i, gauge.head()))

class Result(object):
    def __init__(self, jobpath):
        self._jobpath = jobpath
        self._jobresult = jobresult.JobResult(jobpath)

    @property
    def isEmpty(self):
        log.debug('getProcessTable')
        temp = self._jobresult.getProcessTable()
        if (not temp.isValid) and ('No process found' in temp.message):
            return True
        else:
            return False

    def gaugeset(self, condtable):
        ''' Gauge table containing all the conditions
        '''
        gaugeset = []
        for idx, index in enumerate(condtable.index):
            log.debug('getGaugeTable')
            gauge = self._jobresult.getGaugeTable(int(condtable.ix[index, 'ID']))
            if not gauge.isValid:
                raise RuntimeError('Error in reading gauge file - {}'.format(gauge.message))
            header = '\t'.join([gauge.header[i] for i in range(gauge.getTableColumnNumber())])
            data = ['\t'.join([gauge.data[i][j] for j in range(gauge.getTableColumnNumber())]) for i in range(gauge.getTableRowNumber())]
            gaugestr = '\n'.join([header] + data)
            gauge = pd.read_csv(StringIO(gaugestr), sep='\t')
            gauge['condid'] = int(idx)
            gauge['condwt'] = float(condtable.ix[index, 'Weight'])
            gaugeset.append(gauge)
        gauge = pd.concat(gaugeset, axis=0)
        gname = filter(lambda x: bool(re.match(r'gauge', x, re.IGNORECASE)), gauge.columns)[0]
        gauge[gname] = pd.Series([x.replace('"', QUOTASTR) for x in gauge[gname]], index=gauge.index)
        return gauge.reset_index().set_index([gname, 'condid'])

    def addProcess(self, stage, descr, params, errors):
        '''Args:
        stage: str
        descr: str
        params: dict
        errors: dict of pandas.Series'''
        params = dict(zip(params.keys(), [str(x) for x in params.values()]))
        for k, v in errors.items():
            errors[k] = dict([(x, [str(y) for y in v.ix[x,:].tolist()]) for x in v.index])
        log.debug('addProcess')
        msg = self._jobresult.addProcess(stage, descr, params, errors)
        return msg

    def updateProcess(self, pid, descr, params, errors):
        params = dict(zip(params.keys(), [str(x) for x in params.values()]))
        for k, v in errors.items():
            errors[k] = dict([(x, [str(y) for y in v.ix[x,:].tolist()]) for x in v.index])
        log.debug('updateProcess')
        msg = self._jobresult.updateProcess(pid, descr, params, errors)
        return msg

    def conditiontable(self, pid):
        log.debug('getConditionTable')
        table = self._jobresult.getConditionTable(pid)
        data = []
        for i in range(table.getTableRowNumber()):
            data.append(table.data[i])
        table = pd.DataFrame(data, columns=table.header)
        return table

    def getModelpath(self, pid, condid):
        log.debug('getModelPath')
        modelpath = self._jobresult.getModelPath(int(pid), int(condid))
        if modelpath == '':
            log.error('Fail to get Model path for process {}, condition {}, job {}'.format(pid, condid, self._jobpath))
            raise RuntimeError('Fail to get Model Path')
        return os.path.join(self._jobpath, modelpath)

    def addTermEffective(self, pid, effectivedata):
        resultpath = r'h/cache/dummydb/result/calibrate/job1'
        tree = ET.parse(os.path.join(self._jobpath, resultpath, r'femcalresults.xml'))
        root = tree.getroot()
        for calresult in root.findall('CalResult'):
            if int(calresult.findtext('processid')) == pid:
                idx = calresult.findtext('id')
                effectpath = os.path.join(resultpath, 'termeff{}.txt'.format(idx))
                calresult.find('errorfile').text += r',termeffpath={}'.format(effectpath)
                effectivedata.to_csv(os.path.join(self._jobpath, effectpath), sep=' ')
        tree.write(os.path.join(self._jobpath, resultpath, r'femcalresults.xml'))

    def getProcessTable(self):
        ''' Obtain Process table

        Returns
        -----
        table: pd.DataFrame
            A DataFrame containing process table info
        '''
        table = self._jobresult.getProcessTable()
        if not table.isValid:
            raise RuntimeError('Error in reading process table')

        ''' Convert the internal data structure to pandas DataFrame'''
        header = '\t'.join([table.header[i] for i in range(table.getTableColumnNumber())])
        data = ['\t'.join([table.data[i][j] for j in range(table.getTableColumnNumber())]) for i in range(table.getTableRowNumber())]
        table = '\n'.join([header] + data)
        table = pd.read_csv(StringIO(table), sep='\t')
        return table

    def getGaugeTable(self, condid):
        ''' Obtain gauge table data, not contain results

        Parameters
        -----
        condid: int
            condition id

        Returns
        -----
        gauge: pd.DataFrame
            A dataframe containing gauge table info
        '''
        condid = int(condid)
        gauge = self._jobresult.getGaugeTable(condid)
        if not gauge.isValid:
            raise RuntimeError('Error in reading gauge file - {}'.format(gauge.message))

        ''' Convert the internal data structure to pandas DataFrame'''
        header = '\t'.join([gauge.header[i] for i in range(gauge.getTableColumnNumber())])
        data = ['\t'.join([gauge.data[i][j] for j in range(gauge.getTableColumnNumber())]) for i in range(gauge.getTableRowNumber())]
        gaugestr = '\n'.join([header] + data)
        gauge = pd.read_csv(StringIO(gaugestr), sep='\t')

        gauge['condid'] = condid
        gname = filter(lambda x: bool(re.match(r'gauge', x, re.IGNORECASE)), gauge.columns)[0]
        return gauge.set_index([gname, 'condid'])

    def getGaugeTableWiRes(self, pid, condid):
        ''' Obtain gauge table with result of a process and condition

        Parameters
        -----
        pid: int
            Process id
        condid: int
            condition id

        Returns
        -----
        gauge: pd.DataFrame
            A dataframe containing gauge table info
        '''
        pid, condid = int(pid), int(condid)
        gauge = self._jobresult.getGaugeTableWithResult(pid, condid)
        if not gauge.isValid:
            raise RuntimeError('Error in reading gauge file - {}'.format(gauge.message))

        ''' Convert the internal data structure to pandas DataFrame'''
        header = '\t'.join([gauge.header[i] for i in range(gauge.getTableColumnNumber())])
        data = ['\t'.join([gauge.data[i][j] for j in range(gauge.getTableColumnNumber())]) for i in range(gauge.getTableRowNumber())]
        gaugestr = '\n'.join([header] + data)
        gauge = pd.read_csv(StringIO(gaugestr), sep='\t')

        gauge['condid'] = condid
        gname = filter(lambda x: bool(re.match(r'gauge', x, re.IGNORECASE)), gauge.columns)[0]
        return gauge.set_index([gname, 'condid'])

    def gaugeset(self, condtable):
        ''' Gauge table containing all the conditions
        '''
        gaugeset = []
        for idx in condtable.index:
            condid = condtable.ix[idx, 'ID']
            log.debug('getGaugeTable')
            table = self.getGaugeTable(condid).reset_index().drop(['condid'], axis=1)
            cols = table.columns
            defocus, dose = '#Delta_Focus(nm)', 'Delta_Dose(%)'
            table[defocus] = float(condtable.ix[idx, 'Delta_Focus'])
            table[dose] = float(condtable.ix[idx, 'Delta_Dose'])
            cols = [defocus, dose] + cols.tolist()
            table = table.reindex(columns=cols)
            gaugeset.append(table)
        gaugeset = pd.concat(gaugeset)
        gname = filter(lambda x: bool(re.match(r'gauge', x, re.IGNORECASE)), gaugeset.columns)[0]
        gaugeset = gaugeset.set_index([defocus, dose, gname])
        return gaugeset

    def gaugesetWiRes(self, pid, condtable):
        ''' Gauge table containing all the conditions
        '''
        gaugeset = []
        for idx in condtable.index:
            condid = condtable.ix[idx, 'ID']
            log.debug('getGaugeTable')
            table = self.getGaugeTableWRes(pid, condid).reset_index().drop(['condid'], axis=1)
            cols = table.columns
            defocus, dose = '#Delta_Focus(nm)', 'Delta_Dose(%)'
            table[defocus] = float(condtable.ix[idx, 'Delta_Focus'])
            table[dose] = float(condtable.ix[idx, 'Delta_Dose'])
            cols = [defocus, dose] + cols.tolist()
            table = table.reindex(columns=cols)
            gaugeset.append(table)
        gaugeset = pd.concat(gaugeset)
        gname = filter(lambda x: bool(re.match(r'gauge', x, re.IGNORECASE)), gaugeset.columns)[0]
        gaugeset = gaugeset.set_index([defocus, dose, gname])
        return gaugeset

if __name__ == '__main__':
    jobpath = r'/gpfs/DEV/FEM/peyang/release/E8.0/MOD9944/job2_2D_CD+EP_align_correction_GN'
    from FileUtil import gpfs2WinPath
    jobpath = gpfs2WinPath(jobpath)
    with FemplusJob(jobpath) as femplusjob:
        femresult = Result(jobpath)
        print femresult.getConditionTable()
