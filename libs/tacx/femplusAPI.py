# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 18:08:47 2017

**Dependent to TACX FEM+ python api now, need run by tachyon_python !!!**

This API is covered on top of FEMPlusJob, which provided basic C++ based API.
By arranging the flatten basic level API into object oriented design, we
provide user an easier and clearer interface to handle FEM job.

Class hierarchy describe the relation among all FEM job flow types, each with its
customized methods. The methods are frequently used functions in daily work.
Some complexed components are also abstracted into classes.
See the class hierarchy at :
http://confluence-brion.asml.com/display/~hshi/FEM+Class+Hierarchy+design

FEMPlusJob opens a job, create object in memory.
All get, set methods act on it.
This mechanism can't grantee newest data, because another thread may modify
the job DB or data but the already open object is not aware of it.
To keep working on newest data, we must open the job each time before accessing
data requirement.

This API not cover all aspect of GUI operations. However, the FEMPlusJob API
is more powerful.
Please inherit some of the classes to make extensions.
"""
import shutil
import re
import numpy as np
import pandas as pd
try:
    from cStringIO import StringIO
except ModuleNotFoundError:
    from io import StringIO
from abc import ABCMeta, abstractmethod
import xml.etree.ElementTree as ET
import time
import tempfile

import FEMPlusJob
import jobresult
import mod
import Tac
import FEMSetup
import FEMReview
import ControlCenter

from TachyonJob import Job

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
from MiscUtil import QUOTASTR, Range
from logger import logger
log = logger.getLogger(__name__)

__author__ = "Model PEG"
__credits__ = []
__version__ = "0.0.1"
__status__ = "Development"

'''
# Job Class Registration, please found `FLOW2CLASS` dict in the middle
'''

DF = 'Delta_Focus(nm)'
DD = 'Delta_Dose(%)'

def getFlow(jobpath):
    log.debug('openJob')
    basejob = FEMPlusJob.openJob(jobpath)
    if basejob is None:
        raise IOError('Open job {} failed!'.format(jobpath))
    log.debug('getFemFlowType')
    flowtype = basejob.getFemFlowType()
    #log.debug('releaseJob')
    #FEMPlusJob.releaseJob(basejob)
    return flowtype

def openJob(jobpath):
    ''' The entrance function.
    Automatically choose the right class according to the flowtype

    args:
    -----
    jobpath : str

    returns:
    -----
    job: instance of a child class of FEMJob
    '''
    flowtype = getFlow(jobpath)
    thisname, apiname = FLOW2CLASS.get(flowtype, FLOW2CLASS.values()[0])
    basejob = apiname.openJob(jobpath)
    if basejob is None:
        raise IOError('Open job {} failed!'.format(jobpath))
    job = thisname(basejob)
    return job

def isCheckJob(jobpath):
    flowtype = getFlow(jobpath)
    return flowtype == FEMPlusJob.FLOW_NAME_CHECK

def isCalJob(jobpath):
    flowtype = getFlow(jobpath)
    if flowtype == FEMPlusJob.FLOW_NAME_CAL:
        return True
    else:
        return False

class JobCompoent(object):
    def __init__(self, basejob):
        self._basejob = basejob


class Machine(JobCompoent):
    pass


class Illumination(JobCompoent):
    pass


class Condition(JobCompoent):
    DF = DF
    DD = DD

    @property
    def table(self):
        ''' pandas.DataFrame
        Condition table, containing PW information
        '''
        log.debug('getConditionTable')
        table = self._basejob.getConditionTable()
        temp = [[table.cell(x,y) for y in range(table.columnCount())] for x in range(table.rowCount())]
        return pd.DataFrame(temp, columns=[table.colHeader(x) for x in range(table.columnCount())])

    @property
    def NC(self):
        ''' nominal condition definition:
        delta_focus and delta_dose is both 0
        not considering multi exposure w/o empty role
        if found such condition, return the index of this condition, not the condition id
        if not found such condition, return -1
        This value will be mainly used in linear solver
        '''
        table = self.table
        ncfilter = (table[self.DF] == '0') & (table[self.DD] == '0')
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


class Resist(JobCompoent):
    pass


class CalResist(Resist):
    pass


class ADICalResist(CalResist):
    FEM = 'fem'
    GENERIC = 'generic'
    @property
    def type(self):
        log.debug('getResistModel')
        return self._basejob.getResistModel()

    @type.setter
    def type(self, modeltype):
        log.debug('getResistModel')
        status = self._basejob.setResistModel(modeltype)
        if status != 0:
            raise RuntimeError('Fail to set model type: {}'.format(modeltype))
        log.debug('save')
        self._basejob.save()

    def exportRP(self):
        ''' should have API for this, currently lacking
        '''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rp', prefix='ResistRP', delete=False) as f:
            rstfilepath = f.name
        log.debug('exportResistParams')
        self._basejob.exportResistParams(rstfilepath)
        with open(rstfilepath) as f:
            rststr = f.read()
        try:
            os.remove(rstfilepath)
        except:
            pass
        return rststr

    def importRP(self, rststr):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rp', prefix='ResistRP', delete=False) as f:
            rstfilepath = f.name
            f.write(rststr)
        log.debug('importResistParams')
        status = self._basejob.importResistParams(rstfilepath)
        if status < 0:
            log.warn('Status:{}, Failed to import resist parameter\n {}\n into job {}'.format(status, rststr, self._basejob.getPath()))
        log.debug('save')
        self._basejob.save()
        try:
            os.remove(rstfilepath)
        except:
            pass


class AEICalResist(CalResist):
    pass


class GenResist(Resist):
    pass


class ExpComponents(object):
    __metaclass__ = ABCMeta

    def __init__(self, basejob, rolename):
        self._basejob = basejob
        self._rolename = rolename

    @abstractmethod
    def getAllParams(self):
        return dict()

    @abstractmethod
    def getAllParamsRange(self):
        return dict()

    def __eq__(self, other):
        params1 = self.getAllParams().items()
        params2 = other.getAllParams().items()
        if len(params1) == 0 and len(params2) == 0:
            return True
        return params1 == params2

    def __le__(self, other):
        params1 = self.getAllParamsRange()
        params2 = other.getAllParamsRange()
        boolv = {}
        for k, v in params1.iteritems():
            boolv[k] = (v <= params2.get(k, set()))
        return all(boolv)


class Optical(ExpComponents):
    ILLUMINATIONCLASS = Illumination

    def __init__(self, basejob, rolename):
        self._basejob = basejob
        self._rolename = rolename

    def getAllParams(self):
        params = self.getAllParamsRange()
        return dict([(k, v.min) for k, v in params.iteritems()])

    def getAllParamsRange(self):
        ''' Helper function to collect all the available optical parameters,
        including both fixed and searched parameters

        returns:
        -----
        params : dict
        '''
        log.debug('getOpticalVariables')
        params = self._basejob.getOpticalVariables(self._rolename)
        params = dict([(x.name, Range(x.min, x.max, x.step)) for x in params])
        return params

    def setAllParams(self, paramdict):
        log.debug('getOpticalVariables')
        params = self._basejob.getOpticalVariables(self._rolename)
        for k, p in enumerate(params):
            try:
                pp = paramdict[p.name]
            except KeyError:
                pass
            else:
                params[k].min = pp.min
                params[k].max = pp.max
                params[k].step = pp.step
        log.debug('setOpticalVariables')
        self._basejob.setOpticalVariables(self._rolename, params)
        log.debug('save')
        self._basejob.save()


class FLEX(ExpComponents):
    def __init__(self, basejob, rolename):
        self._basejob = basejob
        self._rolename = rolename

    @property
    def enabled(self):
        pass

    def getAllParams(self):
        ''' Helper function to collect all the available FLEX parameters,
        including both fixed and searched parameters

        returns:
        -----
        params : dict
        '''
        log.debug('getAllFLEXParams')
        params = self._basejob.getAllFLEXParams(self._rolename)
        params = dropNAinDict(dict(params))
        return params

    def getAllParamsRange(self):
        ''' Helper function to collect all the available optical parameters,
        including both fixed and searched parameters

        returns:
        -----
        params : dict
        '''
        params = self.getAllParams()
        params = dict([(k, Range(v, v, v)) for k, v in params.iteritems()])
        return params


class Mask(ExpComponents):
    def __init__(self, basejob, rolename):
        self._basejob = basejob
        self._rolename = rolename

    @staticmethod
    def realimage2transdeg(realimage):
        trans = np.sum(np.power(realimage,2))
        degree = np.rad2deg(np.arctan2(realimage[1], realimage[0]))
        return (trans, degree)

    @property
    def tnps(self):
        ''' In trans degree format'''
        log.debug('getExposureInfo')
        exp = self._basejob.getExposureInfo(self._rolename).ExposureData
        try:
            field = eval(exp.field)
        except SyntaxError:
            try:
                field = eval(exp.fieldRI)
                field = self.realimage2transdeg(field)
            except SyntaxError:
                raise AttributeError('no validate TNP values found')
        polygon = []
        for k in exp.tnps:
            try:
                tnp = eval(k.tnpvalue)
            except SyntaxError:
                try:
                    tnp = eval(k.tnpRIvalue)
                    tnp = self.realimage2transdeg(tnp)
                except SyntaxError:
                    raise AttributeError('no validate TNP values found')
            polygon.append(tnp)
        return (polygon, field)

    def getAllParams(self):
        params = self.getAllParamsRange()
        return dict([(k, v.min) for k, v in params.iteritems()])

    def getAllParamsRange(self):
        ''' Helper function to collect all the available optical parameters,
        including both fixed and searched parameters

        returns:
        -----
        params : dict
        '''
        log.debug('getMaskVariables')
        params = self._basejob.getMaskVariables(self._rolename)
        params = dict([(x.name, Range(x.min, x.max, x.step)) for x in params])
        return params

    def setAllParams(self, paramdict):
        log.debug('getMaskVariables')
        params = self._basejob.getMaskVariables(self._rolename)
        for k, p in enumerate(params):
            try:
                pp = paramdict[p.name]
            except KeyError:
                pass
            else:
                params[k].min = pp.min
                params[k].max = pp.max
                params[k].step = pp.step
        log.debug('setMaskVariables')
        self._basejob.setMaskVariables(self._rolename, params)
        log.debug('save')
        self._basejob.save()


class Exposure(object):
    OPTICALCLASS = Optical
    MASKCLASS = Mask
    FLEXCLASS = FLEX

    def __init__(self, basejob, rolename):
        ''' Containing all information for one exposure
        '''
        self._basejob = basejob
        self.rolename = rolename

    @property
    def optical(self):
        ''' Instance of Optical class to deal with optical settings in an exposure
        '''
        optical = self.OPTICALCLASS(self._basejob, self.rolename)
        return optical

    @property
    def mask(self):
        ''' Instance of Mask class to deal with optical settings in an exposure
        '''
        mask = self.MASKCLASS(self._basejob, self.rolename)
        return mask

    @property
    def flex(self):
        return self.FLEXCLASS(self._basejob, self.rolename)

    @property
    def isBrightField(self):
        log.debug('getExposureInfo')
        exp = self._basejob.getExposureInfo(self.rolename)
        return exp.ExposureData.hasBrightField()

    def __eq__(self, other):
        return ((self.optical == other.optical) and
                (self.mask == other.mask) and
                (self.flex == other.flex))

    def __le__(self, other):
        return ((self.optical <= other.optical) &
                (self.mask <= other.mask) &
                (self.flex <= other.flex))


class FEMJob(Job):
    LOGFILE = 'log.txt'
    datarelpath = r"h/data/dummydb/calibrate/job1"
    resultrelpath = r"h/cache/dummydb/result/calibrate/job1"
    omit_gauge_Cols = ['Error %', 'Model CD', 'Through Cond. Error'] 
    omit_process_Cols = ['Description', 'Process Path'] # 'Process', 
    filekey_GUICol_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'filekey_GUICol.txt')

    def __init__(self, basejob):
        ''' The ancestor class for all flow type FEM jobs
        Some common methods for general FEM jobs are contained
        '''
        self._basejob = basejob
        log.debug('getPath')
        self.jobpath  = self._basejob.getPath()

        super(FEMJob, self).__init__(self.jobpath)
        self.dataabspath = os.path.join(self.jobpath, *(FEMJob.datarelpath.split("/")))
        self.resultabspath = os.path.join(self.jobpath, *(FEMJob.resultrelpath.split("/")))

    def setOption(self, **kwargs):
        for k, v in kwargs.iteritems():
            log.debug('setJobOption')
            self._basejob.setJobOption(str(k), str(v))
        log.debug('save')
        self._basejob.save()

    def getOption(self):
        p = FEMPlusJob.StrStrMap()
        log.debug('getJobOption')
        self._basejob.getJobOptions(p)
        return dict(p)

    def clone(self, newpath):
        if os.path.exists(newpath):
            raise IOError('Folder {} already exist, please delete it or change another one'.format(newpath))
        log.debug('saveAs')
        success = self._basejob.saveAs(newpath)
        if not success:
            raise RuntimeError('Clone job failed! May need to migrate the job first')

    def save(self):
        log.debug('save')
        self._basejob.save()

    def refresh(self):
        self._basejob = FEMJob.openJob(self.jobpath)

    def setLua(self, luatype, luastr):
        job = FEMPlusJob.openJob(self.jobpath)
        job.setLua(luatype, luastr)
        job.save()

    def clearJobResult(self):
        resultdir = os.path.join(self.jobpath, 'h/cache/dummydb/result/calibrate/job1')
        if os.path.exists(resultdir):
            for fid in os.listdir(resultdir):
                fid = os.path.join(resultdir, fid)
                if os.path.isfile(fid):
                    os.unlink(fid)
                elif os.path.isdir(fid):
                    shutil.rmtree(fid)
        log = os.path.join(self.jobpath, LOGFILE)
        if os.path.exists(log):
            os.unlink(log)

    @staticmethod
    def rms(error_, wt_):
        flt = np.where(wt_ > 0)
        error = error_[flt]
        wt = wt_[flt]
        if len(error) == 0:
            return np.nan
        else:
            return np.sqrt(np.sum((np.power(error, 2) * wt))/np.sum(wt))

class ADIJob(FEMJob):
    EXPOSURECLASS = Exposure
    MACHINECLASS = Machine
    DATACLASS = Condition
    RESISTCLASS = Resist

    def __init__(self, *args, **kwargs):
        super(ADIJob, self).__init__(*args, **kwargs)

    @property
    def exposures(self):
        log.debug('getAllOpticalRoles')
        rolenames = self._basejob.getAllOpticalRoles()
        exposures = {}
        for role in rolenames:
            exposures[role] = self.EXPOSURECLASS(self._basejob, role)
        return exposures

    @property
    def data(self):
        return self.DATACLASS(self._basejob)

    @property
    def resist(self):
        return self.RESISTCLASS(self._basejob)

    @property
    def pixel(self):
        log.debug('getSimuGeometry')
        p = self._basejob.getSimuGeometry()
        return p.pixel * 1000

    @property
    def isBrightField(self):
        if all([v.isBrightField for v in self.exposures.values()]):
            return -1
        else:
            return 1

    @property
    def isEUV(self):
        log.debug('isEUV')
        return self._basejob.isEUV()

    def loadModel(self, modelpath, optical=True, resist=True):
        optionstr = []
        if optical:
            optionstr.append('optical')
        if resist:
            optionstr.append('resist')
        log.debug('loadModel')
        self._basejob.loadModel(modelpath, ','.join(optionstr))
        log.debug('save')
        self._basejob.save()

    def opticalEqualTo(self, other):
        ''' All the optical parameters are identical to another job
        '''
        return ((self.exposures.items() == other.exposures.items()) and
                self.data == other.data)

    def opticalIsSubSetof(self, other):
        ''' All the optical parameters are identical to another job
        '''
        return ((self.exposures.items() <= other.exposures.items()) and
                self.data == other.data)


class ADIGenJob(ADIJob):
    RESISTCLASS = GenResist
    pass


class ADICalJob(ADIJob):
    RESISTCLASS = ADICalResist
    DATACLASS = Data

    def __init__(self, *args, **kwargs):
        super(ADICalJob, self).__init__(*args, **kwargs)
        self.result = Result(self.jobpath)


class AutoCalJob(ADIJob):
    pass


class CheckJob(FEMJob):
    DATACLASS = Condition

    def __init__(self, *args, **kwargs):
        super(CheckJob, self).__init__(*args, **kwargs)
        self.result = Result(self.jobpath)


class AEIJob(FEMJob):
    pass


class AEIGenJob(AEIJob):
    pass


class AEICalJob(AEIJob):
    pass


class M3DLibGenJob(FEMJob):
    pass


class W3DLibGenJob(FEMJob):
    pass


class FlareGenJob(FEMJob):
    pass


''' There is one class for each flow type
'''
FLOW2CLASS = {FEMPlusJob.FLOW_NAME_CAL: (ADICalJob, FEMPlusJob.ModelCalibration),
              FEMPlusJob.FLOW_NAME_GEN: (ADIGenJob, FEMPlusJob.ModelGeneration),
              FEMPlusJob.FLOW_NAME_AUTOCAL: (AutoCalJob, FEMPlusJob.AutoCalibration),
              FEMPlusJob.FLOW_NAME_CHECK: (CheckJob, FEMPlusJob.ModelCheck),
              FEMPlusJob.FLOW_NAME_AEICAL: (AEICalJob, FEMPlusJob.AEIModelCal),
              FEMPlusJob.FLOW_NAME_AEIGEN: (AEIGenJob, FEMPlusJob.AEIModelGen),
              FEMPlusJob.FLOW_NAME_M3D: (M3DLibGenJob, FEMPlusJob.M3dLibGen),
              FEMPlusJob.FLOW_NAME_W3D: (W3DLibGenJob, FEMPlusJob.W3dLibGen),
            } # GUI-25367, small typo, will change back after fixing

def dropNAinDict(mydict):
    ''' Filter out the notFoundVal in some return values by basic API.
    '''
    def filterfun(x):
        return FEMPlusJob.FEMPlusJob_isNotFoundVal(x[1])
    try:
        nakeys = zip(*filter(filterfun, mydict.iteritems()))[0]
        for k in nakeys:
            mydict.pop(k)
    except IndexError:
        pass
    return mydict


class Result(object):
    DF = DF
    DD = DD

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
        for k, v in errors.iteritems():
            errors[k] = dict([(x, [str(y) for y in v.ix[x,:].tolist()]) for x in v.index])
        log.debug('addProcess')
        msg = self._jobresult.addProcess(stage, descr, params, errors)
        return msg

    def updateProcess(self, pid, descr, params, errors):
        params = dict(zip(params.keys(), [str(x) for x in params.values()]))
        for k, v in errors.iteritems():
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

    def getGaugeTableWRes(self, pid, condid):
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
            table[self.DF] = float(condtable.ix[idx, self.DF])
            table[self.DD] = float(condtable.ix[idx, self.DD])
            cols = [self.DF, self.DD] + cols.tolist()
            table = table.reindex(columns=cols)
            gaugeset.append(table)
        gaugeset = pd.concat(gaugeset)
        gname = filter(lambda x: bool(re.match(r'gauge', x, re.IGNORECASE)), gaugeset.columns)[0]
        gaugeset = gaugeset.set_index([self.DF, self.DD, gname])
        return gaugeset

    def gaugesetWRes(self, pid, condtable):
        ''' Gauge table containing all the conditions
        '''
        gaugeset = []
        for idx in condtable.index:
            condid = condtable.ix[idx, 'ID']
            log.debug('getGaugeTable')
            table = self.getGaugeTableWRes(pid, condid).reset_index().drop(['condid'], axis=1)
            cols = table.columns
            table[self.DF] = float(condtable.ix[idx, self.DF])
            table[self.DD] = float(condtable.ix[idx, self.DD])
            cols = [self.DF, self.DD] + cols.tolist()
            table = table.reindex(columns=cols)
            gaugeset.append(table)
        gaugeset = pd.concat(gaugeset)
        gname = filter(lambda x: bool(re.match(r'gauge', x, re.IGNORECASE)), gaugeset.columns)[0]
        gaugeset = gaugeset.set_index([self.DF, self.DD, gname])
        return gaugeset

    def gaugesetWResCondId(self, pid, condtable, result_only=False, extra_cols=None):
        ''' Gauge table containing all the conditions
        '''
        gaugeset = []
        for idx in condtable.index:
            condid = condtable.ix[idx, 'ID']
            log.debug('getGaugeTable')
            table = self.getGaugeTableWRes(pid, condid).reset_index()
            cols = table.columns
            table[self.DF] = float(condtable.ix[idx, self.DF])
            table[self.DD] = float(condtable.ix[idx, self.DD])
            if result_only:
                idx = cols.tolist().index("Model Error")
                gname = filter(lambda x: bool(re.match(r'gauge', x, re.IGNORECASE)), table.columns)[0]
                extra_cols = [] if extra_cols is None else extra_cols
                cols = [gname, 'condid'] + extra_cols + cols.tolist()[idx:]
            else:
                cols = [self.DF, self.DD, 'condid'] + cols.tolist()
            table = table.reindex(columns=cols)
            gaugeset.append(table)
        gaugeset = pd.concat(gaugeset)
        gname = filter(lambda x: bool(re.match(r'gauge', x, re.IGNORECASE)), gaugeset.columns)[0]
        gaugeset = gaugeset.set_index([gname, "condid"])
        return gaugeset


class Model(object):
    def __init__(self, path):
        self._model = mod.TFlexProcess(path)

    @property
    def resistParams(self):
        return dict(self._model.getResist().params.items())


class TacJob(object):
    def __init__(self, jobpath):
        log.debug('getTac')
        _TC = Tac.getTac()
        self._SETUP = _TC.getFEMSetup()
        self._REVIEW = _TC.getFEMReview()
        self._CC = _TC.getControlCenter()
        self.jobpath = jobpath
        log.debug('getTac done')

    @property
    def isSetupOpen(self):
        log.debug('_SETUP.getAllJobs')
        joblist = self._SETUP.getAllJobs()
        try:
            return self.jobpath in joblist
        except TypeError:
            return True

    @property
    def isReviewOpen(self):
        log.debug('_REVIEW.getAllJobs')
        joblist = self._REVIEW.getAllJobs()
        try:
            return self.jobpath in joblist
        except TypeError:
            return True

    def openSetup(self):
        log.debug('_SETUP.openJob')
        status = self._SETUP.openJob(self.jobpath)
        if not status:
            raise RuntimeError('Open job setup failed, please check the job path: {}'.format(self.jobpath))

    def openReview(self):
        log.debug('_SETUP.openFEMReview')
        status = self._REVIEW.openFEMReview(self.jobpath)
        if not status:
            raise RuntimeError('Open job review failed, please check the job path: {}'.format(self.jobpath))

    def submit(self):
        if self.isSetupOpen:
            log.debug('_SETUP.submit')
            self._SETUP.submitJob(self.jobpath)
        else:
            log.warn('Job not opened, submit failed')

    def saveSetup(self):
        if self.isSetupOpen:
            log.debug('_SETUP.saveSetup')
            self._SETUP.saveSetup(self.jobpath)
        else:
            log.warn('Job not opend, save setup failed')

    def saveAs(self, newpath):
        if self.isSetupOpen:
            log.debug('_SETUP.saveSetupAs')
            return self._SETUP.saveSetupAs(newpath, self.jobpath)
        else:
            log.warn('Job not opend, save setup failed')

    def refreshSetup(self):
        if self.isSetupOpen:
            log.debug('_SETUP.refresh')
            self._SETUP.refresh(self.jobpath)
        else:
            log.warn('Job not opend, refresh setup failed')

    def refreshReview(self):
        if self.isReviewOpen:
            log.debug('_REVIEW.refresh')
            self._REVIEW.refresh(self.jobpath)
        else:
            log.warn('Job not opend, refresh review failed')

    def refreshProcess(self):
        if self.isReviewOpen:
            log.debug('_REVIEW.updateProcess')
            self._REVIEW.updateProcess(self.jobpath)
        else:
            log.warn('Job not opend, refresh review failed')

    def getHighlightedProcessID(self):
        if self.isReviewOpen:
            log.debug('_REVIEW.getSelectedProcessId')
            pid = self._REVIEW.getSelectedProcessId(self.jobpath)
            if pid < 0:
                return None
            else:
                return pid
        else:
            return None

    def getHighlightedConditionID(self):
        if self.isReviewOpen:
            log.debug('_REVIEW.getSelectedConditionId')
            condid = self._REVIEW.getSelectedConditionId(self.jobpath)
            if condid < 0:
                return None
            else:
                return condid
        else:
            return None

    def getFiteredGaugeList(self):
        if self.isReviewOpen:
            log.debug('_REVIEW.getFilteredGaugeList')
            return self._REVIEW.getFilteredGaugeList(self.jobpath).split()
        else:
            return None

    def addTask(self, taskname, taskcommand, postscript=''):
        taskcommand = str(taskcommand)
        log.debug('_CC.addTask')
        return self._CC.addTask(taskname, taskcommand, postscript)

    def stopTask(self, taskID):
        log.debug('_CC.stopTask')
        return self._CC.stopTask(taskID)

if __name__ == '__main__':
    from FileUtil import gpfs2WinPath
    jobpath = r"/gpfs/DEV/FEM/peyang/release/E8.0/MOD9944/job2_2D_CD+EP_align_correction_GN_clone"
    processid = 1037412475
    jobpath = gpfs2WinPath(jobpath)
    job = Result(jobpath)
    condtable = job.conditiontable(processid)
    # print(job.gaugesetWRes(processid, condtable).head())
    
    extra_cols = ['cost_wt', 'GaugeClusterId']
    resultSet = job.gaugesetWResCondId(processid, condtable, True, extra_cols)
    
    oriErrKey = 'Model Error'
    newErrKey = 'cluster_shiftd_error' if 'cluster_shiftd_error' in resultSet.columns else 'cluster_shifted_error'
    newErrFunc = lambda s: s[oriErrKey] if np.isnan(s[newErrKey]) or s[newErrKey] == 0 else s[newErrKey]
    resultSet.loc[:, newErrKey] = resultSet.apply(newErrFunc, axis=1)
    oriRms = FEMJob.rms(resultSet.loc[:, oriErrKey].values, resultSet.loc[:, 'cost_wt'].values)
    newRms = FEMJob.rms(resultSet.loc[:, newErrKey].values, resultSet.loc[:, 'cost_wt'].values)
    descr = "original RMS {:.4f} cluster shifted RMS {:.4f}".format(oriRms, newRms)
    # descr = re.sub('\s', '_', descr)
    print(descr)

    try:
        processtable = job.getProcessTable().astype(str)
        curprocess = processtable.loc[processtable.Process==str(processid), :].reset_index(drop=True)
        params = curprocess.iloc[0].to_dict()
    except:
        params = {"(rms)": str(oriRms)}
    params["(cluster_shifted_rms)"] = str(newRms)
    # params["Description"] = descr
    print(params)
    for key in FEMJob.omit_process_Cols:
        try:
            params.pop(key, None)
        except:
            pass

    filekey_GUICol_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "filekey_GUICol.txt")
    filekey_GUICol = pd.read_csv(filekey_GUICol_path, sep='\t')

    print(resultSet.columns)
    results = {}
    for col in resultSet.columns:
        if col not in extra_cols + FEMJob.omit_gauge_Cols:
            try:
                filekey = filekey_GUICol.loc[filekey_GUICol['GUI Col']==col, 'filekey'].values[0]
            except:
                filekey = col+"_result"
            results[filekey] = resultSet.loc[:, col].unstack()
    print(results.keys())

    print(job.updateProcess(processid, descr, params, results))
