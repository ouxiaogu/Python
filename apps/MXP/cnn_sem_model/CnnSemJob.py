# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-08-31 10:53:39

CNN SEM model Job flow

Last Modified by: ouxiaogu
"""

import os.path
import time
import xml.etree.ElementTree as ET
import re
from collections import OrderedDict
import numpy as np
import glob
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import pandas as pd
from dataset import DataSet, load_image, centered_norm
from convNN import ConvNN
import argparse

import logger
log = logger.setup("MXPJob", "debug")

MXP_XML_TAGS = ["inxml", "outxml"]
MXP_RESULT_OPTIONS = ["occfs", "osumccfs", "osumkpis"]
USAGE_TYPES = ['training', 'validation', 'test']
PATTERN_DEFAULT_ATTR =\
'''name costwt    usage   srcfile    tgtfile  imgpixel   offset_x    offset_y
1   1 training   1_se.bmp    1_image.bmp    1   0   0'''
STAGE_REGISTER_TABLE = {
    'init': 'InitStage',
    'DLSEMCalibration': 'CSemCalStage',
    'DLSEMApply': 'CSemApplyStage'
}

def parseText(text_):
    """
    Returns
    -------
    text_:  the conversion relationship
        string  =>  string
        numeric =>  int or float, decided by whether text_ contains '.'
    """
    if text_ is None:
        return ""
    try: # numeric
        try:
            if re.match(r"^[\n\r\s]+", text_) is not None:
                raise ValueError("Input text is empty\n")
        except TypeError:
            raise TypeError("error occurs when parse text %s\n" % text_)
        if '.' in text_: #float
            text_ = float(text_)
        else: # int
            text_ = int(text_)
    except ValueError: # string
        pass
    return text_

def getElemText(elem_, defaultval=""):
    """
    Parse text of xml Elem

    Parameters
    ----------
    elem_:  xml Elem object,
        if elem_ node has children, raise ValueError;
        else return parsed text
    """
    text_ = elem_.text
    if text_ is None:
        return defaultval
    else:
        return parseText(text_)

def getConfigData(node, key, defaultval=""):
    elem = node.find(key)
    ret = None;
    try:
        ret = getElemText(elem, defaultval)
    except:
        ret = defaultval
    return ret

def getGlobalConfigData(gcf, cf, node, defaultval=""):
    lval = getConfigData(cf, node, defaultval)
    if lval == defaultval:
        return getConfigData(gcf, node, defaultval)
    else:
        return lval

def getConfigMap(elem, trim=False, precision=3):
    """
    Parse the tag & text of all depth=1 children node into a map.

    Example
    -------
    >>> elem
    <pattern>
        <kpi>0.741096070383657</kpi>
        <name>13</name>
    </pattern>
    >>> getConfigMap(elem)
    {'kpi': 0.741, 'name': 13}
    """
    rst = {}
    for item in list(elem):
        if len(item) > 0:
            continue
        rst[item.tag] = getElemText(item)
    return rst

def getRecurConfigMap(elem):
    """
    Recursively parse the tag & text of all children nodes into a map.
    """
    rst = {}
    for item in list(elem):
        if len(item) > 0:
            rst[item.tag] = getRecurConfigMap(item)
        else:
            rst[item.tag] = getElemText(item)
    return rst

def getRecurConfigVec(elem):
    """
    Recursively parse the tag & text of all children nodes into a vector.
    Vector elements are KW cells with len=2
    """
    rst = []
    for item in list(elem):
        if len(item) > 0:
            rst.append((item.tag, getRecurConfigVec(item)))
        else:
            rst.append((item.tag, getElemText(item)))
    return rst

def getConfigMapList(node, pattern):
    """
    For all the items match `pattern` under current node(depth=1),
    based on the result of getConfigMap, output a list of KWs map"""
    rst = []
    if not pattern.startswith("."): # depth=1
        pattern = "."+pattern
        log.debug("add '.' before pattern to only search depth=1: %s", pattern)
    for child in node.findall(pattern):
        curMap = getConfigMap(child)
        rst.append(curMap)
    return rst

def dfFromConfigMapList(node, pattern):
    return pd.DataFrame(getConfigMapList(node, pattern))

def dfRowToXmlStream(row):
    """output DataFrame row into a xml pattern node"""
    indent = '\t\t'
    xml = [indent+'<pattern>']
    for field in row.index:
        xml.append('{}\t<{}>{}</{}>'.format(indent, field, row[field], field))
    xml.append(indent+'</pattern>')
    return '\n'.join(xml)

def dfToXmlStream(df):
    """output pattern Dataframe into config stream"""
    xml = ['<root>']
    xml.append('\t<result>')
    xml.extend(df.apply(dfRowToXmlStream, axis=1))
    xml.append('\t</result>')
    xml.append('</root>')
    return '\n'.join(xml)

class Job(object):
    """organize CNN tasks as a Job, like tachyon"""
    def __init__(self, jobpath):
        self.__buildjob(jobpath)

    def __buildjob(self, jobpath):
        if not os.path.exists(jobpath):
            e = "Job not exists at: {}\n".format(jobpath)
            raise IOError(e)
        self.jobpath = os.path.abspath(jobpath)

        # jobxml = os.path.join(self.jobpath, 'jobinfo', 'job.xml') # tachyon job
        jobxml = os.path.join(self.jobpath, 'job.xml')
        if not os.path.exists(jobxml):
            e = "Job xml not exists at: {}".format(jobpath)
            self.jobxml = None
        self.jobxml = jobxml

    def checkJobXml(self):
        if self.jobxml is None:
            e = "Error occurs when parsing job xml: \n".format(self.jobxml)
            raise IOError(e)
        return True

class MXPJob(Job):
    """docstring for MXPJob"""
    def __init__(self, jobpath):
        super(MXPJob, self).__init__(jobpath)
        self.__buildjob()

    def __buildjob(self):
        self.datarelpath = r"data"
        self.resultrelpath = r"result"
        self.resultabspath = os.path.join(self.jobpath, self.resultrelpath)
        if not os.path.exists(self.resultabspath):
            os.mkdir(self.resultabspath)
        self.mxpxml = self.jobxml

        mxproot = None
        try:
            mxproot = ET.parse(self.mxpxml).getroot().find(".MXP")
        except KeyError:
            raise KeyError("MXP tag not find int xml: %s\n" % self.mxpxml)
        self.mxproot = mxproot

        self.getmxpCfgMap()

    def resultAbsPath(self, basename):
        return os.path.join(self.jobpath, self.resultrelpath, basename)

    def dataAbsPath(self, basename):
        return os.path.join(self.jobpath, self.datarelpath, basename)

    def getEnableRange(self):
        """
        Enable range in MXP job option

        Returns
        -------
        range_: list of len=2
            [-1, -1]: means all the stages are enabled
            [min, max]:stage between [min max] are enabled
        """
        range_ = [-1, -1]
        try:
            # options = self.getOption()
            # range_ = options["enable"]
            range_ = self.mxproot.find('./global/options/enable').text
            range_ = list(map(int, range_.split('-')))
        except:
            pass
        return range_

    def getStageConfig(self, stage):
        if not isinstance(stage, tuple) and len(stage) !=2:
            raise ValueError("Wrong input: %s, please input MXP stage name together with enable number, e.g., (Average, 300)\n" % str(stage_))
        stagename_, enable_ = stage
        found = False
        for item in list(self.mxproot):
            stagename = item.tag
            enable = ""
            try:
                enable = getElemText(item.find(".enable"))
            except:
                pass
            if stagename == stagename_ and enable==enable_:
                found = True
                return item
        if not found:
            return None

    def getmxpCfgMap(self):
        """OrderedDict {(stagename, enable): dict }"""

        mxpCfgMap = OrderedDict()
        for item in list(self.mxproot):
            stagename = item.tag
            enable = ""
            try:
                enable = getElemText(item.find(".enable"))
            except:
                pass
            cfgMap = getRecurConfigMap(item)
            mxpCfgMap[(stagename, enable)] = cfgMap
        self.mxpCfgMap = mxpCfgMap
        return mxpCfgMap

    def getAllMxpStages(self, enabled=True):
        """
        Returns
        -------
        stages : sorted list by enable number
            items like [(stagename, enable), ...], ascending order
        enabled : bool
            if True, just get the enabled stages; if False, get all

        Example
        -------
            [('D2DBAlignment', 500), ('ResistModelCheck', 510),
             ('ImageD2DBAlignment', 600), ('ResistModelCheck', 610)]
        """
        range_ = self.getEnableRange()
        if set(range_) == set([-1, -1]):
            enabled = False
        inRange = lambda x: True if x >= range_[0] and x <= range_[1] else False
        stages = []
        for key, stagectx in self.mxpCfgMap.items():
            stagename, enable_ = key
            if type(enable_) == str:
                continue
            if enabled and not inRange(enable_):
                continue
            stages.append((stagename, enable_))

        # bubble sort
        stagenum = len(stages)
        swaped = True
        for i in range(stagenum):
            if not swaped:
                break
            swaped = False
            for j in range(stagenum-1, 0, -1): # j <- [1, N-1]
                if stages[j][1] < stages[j-1][1]: #swap descending enable
                    tmp = stages[j]
                    stages[j]  = stages[j-1]
                    stages[j-1] = tmp
                    swaped = True
        self.stages = stages
        return stages

    def getAllStageIOFiles(self):
        """get the relative path of all stage xml files
        For both inxml and outxml"""
        stages = self.getAllMxpStages() if not hasattr(self, "stages") else self.stages
        stageIOFiles = OrderedDict()
        for key in stages:
            if not key in stageIOFiles:
                stageIOFiles[key] = OrderedDict()
            for tag in MXP_XML_TAGS:
                try:
                    stageIOFiles[key][tag] = self.mxpCfgMap[key][tag]
                except KeyError:
                    pass
                except TypeError:
                    # log.debug("TypeError, key = %s, tag = %s\n" % (str(key), tag))
                    pass
        self.stageIOFiles = stageIOFiles
        return stageIOFiles

    def getStageIOFile(self, stage=None, stagename=None, enable=None, option="outxml"):
        """get the relative path of stage xml file, default is outxml"""
        stageIOFiles = self.getAllStageIOFiles() if not hasattr(self, 'stageIOFiles') else self.stageIOFiles
        stagenames, enables = zip(*stageIOFiles.keys())
        stage_ = stage
        if stagename is not None and enable is not None:
            stage_ = (stagename, enable)
        elif enable is not None:
            try:
                ix = list(enables).index(enable)
                stage_ = (stagenames[ix], enable)
            except:
                print(stagenames, enables)
                raise IndexError("Input enable number %s not in Job's enable list: %s" % (enable, str(enables)))
        else:
            if isinstance(stage_, str):
                m = re.search("(\d+)$", stage_)
                if m:
                    log.debug("re search result: %s" % str(m.groups()))
                    enable = m.groups()[0]
                    ix = stage_.find(enable)
                    stagename = stage_[:ix]
                    enable = int(enable)
                    log.debug("stagename: %s" % stagename)
                    log.debug("enable: %d" % enable)
                    stage_ = (stagename, enable)
            elif isinstance(stage_, tuple):
                pass
            else:
                raise ValueError("Wrong input: %s, please input MXP stage name together with enable number, e.g., Average300\n" % str(stage_))

        if stage_ not in stageIOFiles:
            raise KeyError("Input stage %s not in: %s" % (stage_, str(stageIOFiles.keys())))
        if option not in MXP_XML_TAGS:
            raise KeyError("Input option %s not in: %s" % (option, str(MXP_XML_TAGS)))
        xmlfile = stageIOFiles[stage_][option]
        # xmlfile = self.dataAbsPath(xmlfile) if option == MXP_XML_TAGS[0] else self.resultAbsPath(xmlfile)
        xmlfile = self.resultAbsPath(xmlfile) # init stage without inxml, other stages inxml and outxml are all in result folder
        return xmlfile

    def getStageOut(self, stage):
        """Refer to MXPStageXml.getoccfs"""
        return self.getStageResultFactory(stage, "occfs")

    def getAllStageOut(self):
        """occf results of all the stage"""
        stageIOFiles = self.getAllStageIOFiles()
        dfs = OrderedDict()
        for stagename in stageIOFiles.keys():
            dfs[stagename] = self.getStageOut(self, stagename)
        return dfs

    def getStageSummary(self, stage):
        """Refer to MXPStageXml.getosumccfs"""
        return self.getStageResultFactory(stage, "osumccfs")

    def getStageSummaryKpi(self, stage):
        """Refer to MXPStageXml.getosumkpis"""
        return self.getStageResultFactory(stage, "osumkpis")

    def getStageResultFactory(self, stage, option="occfs"):
        xmlfile = self.getStageIOFile(stage)
        xmlfile = self.resultAbsPath(xmlfile)
        parser = MXPStageXml(xmlfile)

        if option == "occfs":
            return parser.getoccfs()
        elif option == "osumccfs":
            return parser.getosumccfs()
        elif option == "osumkpis":
            return parser.getosumkpis()
        else:
            raise KeyError("Input MXP result option %s not in: %s" % (option, str(MXP_RESULT_OPTIONS)))

    def run(self):
        allstages = self.getAllMxpStages()
        gcf = self.mxproot.find('.global')
        for stage in allstages:
            stagename, enablenum = stage
            log.info("Stage %s%d starts\n" % (stagename, enablenum))
            df = None
            if stagename != "init":
                inxmlfile = self.getStageIOFile(stage, option=MXP_XML_TAGS[0])
                stagexml = MXPStageXml(inxmlfile, option=MXP_XML_TAGS[0])
                df = stagexml.geticcfs()
            cf = self.getStageConfig(stage)
            stagepath = self.resultAbsPath('{}{}'.format(stagename, enablenum))
            curstage = eval(STAGE_REGISTER_TABLE[stagename])(gcf, cf, df, stagepath)
            curstage.run()
            outxmlfile = self.getStageIOFile(stage, option="outxml")
            curstage.save(outxmlfile)
            log.info("Stage %s%d successfully finished\n" % (stagename, enablenum))

class MXPStageXml(object):
    """docstring for MXPStageXml"""
    def __init__(self, xmlfile, option=MXP_XML_TAGS[1]):
        super(MXPStageXml, self).__init__()
        self.xmlfile = xmlfile
        self.option = option
        self.buildschema()

    def buildschema(self):
        if self.option==MXP_XML_TAGS[0]:
            self.geticf()
        elif self.option==MXP_XML_TAGS[1]:
            self.getocf()
            self.getosumcf()

    def geticf(self):
        try:
            self.icf = ET.parse(self.xmlfile).getroot().find("result")
            log.debug("icf tag: "+self.icf.tag)
        except AttributeError:
            raise AttributeError("%s has no attribute 'tag'" % self.icf)
        return self.icf

    def getocf(self):
        try:
            self.ocf = ET.parse(self.xmlfile).getroot().find("result")
            log.debug("ocf tag: "+self.ocf.tag)
        except AttributeError:
            raise AttributeError("%s has no attribute 'tag'" % self.ocf)

    def getosumcf(self):
        self.osumcf = self.ocf.find("summary")
        try:
            log.debug("osumcf tag: "+self.osumcf.tag)
        except:
            log.debug("No summary tag in input xml： %s" % self.xmlfile)
            pass

    def geticcfs(self):
        return dfFromConfigMapList(self.icf, ".pattern")

    def getoccfs(self):
        """
        Returns
        -------
        ret : DataFrame object
            all the pattern's results in xmlfile
        """
        return dfFromConfigMapList(self.ocf, ".pattern")

    def getosumccfs(self):
        """
        Returns
        -------
        ret : DataFrame object
            all the pattern's summary in xmlfile"""
        return dfFromConfigMapList(self.osumcf, ".pattern")

    def getosumkpis(self):
        """
        Returns
        -------
        ret : DataFrame object\
            all the pattern's KPIsummary in xmlfile"""
        return dfFromConfigMapList(self.osumcf, ".pattern/KPI")

class MxpStage(object):
    """docstring for MxpStage"""
    def __init__(self, gcf, cf, df, stagepath):
        self.d_gcf = gcf
        self.d_cf = cf
        self.d_df = df
        if not os.path.exists(stagepath):
            os.mkdir(stagepath)
        self.stagepath = stagepath
        self.jobresultpath = os.path.dirname(stagepath)
        self.stagename = os.path.basename(stagepath)

    def save(self, path):
        ss = dfToXmlStream(self.d_df)
        log.debug("Result save path: %s\n" % (path))
        with open(path, 'w') as fout:
            fout.write(ss)
        log.info("Result saved at %s\n" % (path))

class InitStage(MxpStage):
    def run(self):
        dataDir = getConfigData(self.d_cf, "data_dir", "")
        folderflt = getConfigData(self.d_cf, ".filter/folder", "*")
        srcfileflt = getConfigData(self.d_cf, ".filter/srcfile", "*")
        tgtfileflt = getConfigData(self.d_cf, ".filter/tgtfile", "*")
        doshuffle = getConfigData(self.d_cf, ".shuffle", 0)
        divide_rule = getConfigData(self.d_cf, ".divide_rule", "70:20:10")
        divide_rule = list(map(int, divide_rule.split(":")))
        if sum(divide_rule) != 100:
            log.warn("Warning, sum of training set, validation set and test set should be 100, input is %s, use [70, 20, 10] instead." % ':'.join(divide_rule))

        if not os.path.exists(dataDir):
            raise IOError("data_dir not exists at: {}".format(dataDir))
        log.info('Going to glob dataset')
        log.info('Now going to read path for both input and target images')
        srcpathex = os.path.join(dataDir, folderflt, srcfileflt)
        tgtpathex = os.path.join(dataDir, folderflt, tgtfileflt)
        srcfiles = glob.glob(srcpathex)
        tgtfiles = glob.glob(tgtpathex)
        if len(srcfiles) != len(tgtfiles):
            raise ValueError("%s, number of source files is not equal with target files', %d != %d\n", len(srcfiles), len(tgtfiles))
        nsamples = len(srcfiles)
        basename = list(map(os.path.basename, (map(os.path.dirname, srcfiles))))
        DATA = StringIO(PATTERN_DEFAULT_ATTR)
        defaultdf = pd.read_csv(DATA, sep="\s+")
        colnames = defaultdf.columns
        defaultrow = defaultdf.as_matrix()
        dataset = np.tile(defaultrow, (nsamples, 1))
        df = pd.DataFrame(dataset, columns=colnames)
        df = df.assign(name=basename, srcfile=srcfiles, tgtfile=tgtfiles)
        if doshuffle > 0:
            df = df.sample(frac=1).reset_index(drop=True) # shuffle

        # assign usage
        divides = [max(1, int(d/100.*nsamples)) for d in divide_rule]
        divides[-1] = nsamples - sum(divides[:-1])
        assert(all(np.array(divides)>=0))
        divides = np.cumsum(divides)
        log.debug("divides: {}".format(divides))
        usages = df.loc[:, 'usage'].values
        usages[:divides[0]] = USAGE_TYPES[0]
        usages[divides[0]:divides[1]] = USAGE_TYPES[1]
        usages[divides[1]:] = USAGE_TYPES[2]
        df = df.assign(usage=usages)

        # restore df
        self.d_df = df

class CSemCalStage(MxpStage):
    def loadDataSet(self):
        nsamples = len(self.d_df)
        imgsize = getGlobalConfigData(self.d_gcf, self.d_cf, ".imgsize", 128)

        shape_order = getConfigData(self.d_cf, ".shape_order", "nchw")
        self.imgsize = imgsize
        if shape_order == "nchw":
            self.datashape = (nsamples, 1, self.imgsize, self.imgsize)
        else: # "nhwc"
            self.datashape = (nsamples, self.imgsize, self.imgsize, 1)

        train_flt = (self.d_df.usage==USAGE_TYPES[0])
        train_input_images_file_name = self.d_df.loc[train_flt, 'srcfile'].values
        train_target_images_file_name = self.d_df.loc[train_flt, 'tgtfile'].values
        train_input_images = np.array([load_image(path, self.imgsize) for path in train_input_images_file_name])
        train_target_images = np.array([load_image(path, self.imgsize) for path in train_target_images_file_name])

        valid_flt = (self.d_df.usage==USAGE_TYPES[1])
        valid_input_images_file_name = self.d_df.loc[valid_flt, 'srcfile'].values
        valid_target_images_file_name = self.d_df.loc[valid_flt, 'tgtfile'].values
        valid_input_images = np.array([load_image(path, self.imgsize) for path in valid_input_images_file_name])
        valid_target_images = np.array([load_image(path, self.imgsize) for path in valid_target_images_file_name])

        doCenteredNorm = getGlobalConfigData(self.d_gcf, self.d_cf, 'centered_normalize_X', 0)
        if doCenteredNorm > 0:
            centered_norm(train_input_images)
            centered_norm(valid_input_images)
        log.debug("X train shape {} {}".format(train_input_images.shape, train_input_images[0].shape))
        self.train = DataSet(train_input_images, train_target_images, train_input_images_file_name, train_target_images_file_name)
        self.valid = DataSet(valid_input_images, valid_target_images, valid_input_images_file_name, valid_target_images_file_name)

        log.info("Complete reading input data")
        log.info("Number of files in Training-set:\t\t{}".format(len(train_input_images_file_name)))
        log.info("Number of files in Validation-set:\t{}".format(len(valid_input_images_file_name)))

    def run(self):
        learning_rate = getConfigData(self.d_cf, ".learning_rate", 1e-4)
        batchsize = getConfigData(self.d_cf, ".batchsize", 8)
        epochs = getConfigData(self.d_cf, ".epochs", 2)
        random_seed = getGlobalConfigData(self.d_gcf, self.d_cf, '.random_seed', 123)
        device = getGlobalConfigData(self.d_gcf, self.d_cf, '.device', 'cpu')
        dataformat = 'channels_last' if device.lower() == 'cpu' else 'channels_first'

        self.loadDataSet()
        X_train, y_train = self.train.input_images, self.train.target_images
        X_valid, y_valid = self.valid.input_images, self.valid.target_images
        log.info('Training:\t{}\t{}'.format(str(X_train.shape), str(y_train.shape)))
        log.info('Validation:\t{}\t{}'.format(str(X_valid.shape), str(y_valid.shape)))
        cnn = ConvNN(batchsize=batchsize, epochs=epochs, learning_rate=learning_rate, random_seed=123, imgsize=self.imgsize, data_format = dataformat)
        kwargs = {'stagepath': self.stagepath}
        cnn.train(training_set=(X_train, y_train),
                  validation_set=(X_valid, y_valid), **kwargs)

        # save cnn model
        self.modelpath= os.path.join(self.stagepath, 'tflayers-model')
        cnn.save(path=self.modelpath, epoch=epochs)

        # compute error for all train/validation data
        rmses = []
        input_flt = (self.d_df.usage!=USAGE_TYPES[2])
        input_srcfiles = self.d_df.loc[input_flt, 'srcfile'].values
        input_tgtfiles = self.d_df.loc[input_flt, 'tgtfile'].values
        for i in range(len(input_srcfiles)):
            imgs = [[load_image(input_srcfiles[i], self.imgsize)], [load_image(input_tgtfiles[i], self.imgsize)]]
            rmses.append(cnn.model_error(*imgs))
        self.d_df.loc[input_flt, 'rms'] = rmses

    def save(self, path):
        # save into xml
        xml = ['<root>']
        xml.append('\t<result>')
        xml.append('\t\t<model>{}</model>'.format(self.modelpath))
        xml.extend(self.d_df.apply(dfRowToXmlStream, axis=1))
        xml.append('\t</result>')
        xml.append('</root>')
        ss = '\n'.join(xml)
        with open(path, 'w') as fout:
            fout.write(ss)
        log.info("Result saved at %s" % (path))

class CSemApplyStage(MxpStage):
    def loadDataSet(self):
        imgsize = getGlobalConfigData(self.d_gcf, self.d_cf, ".imgsize", 128)
        self.imgsize = imgsize
        test_flt = (self.d_df.usage==USAGE_TYPES[2])
        if len(test_flt.nonzero()[0]) == 0:
            frac = 0.2
            nsamples = len(self.d_df)
            testnum = max(1, frac*nsamples)
            log.warning("{} not any pattern labeled as 'test' in the dataset, randomly select data ~frac={}".format(self.stagename, frac))
            test_flt = self.d_df.usage.isin(USAGE_TYPES).sample(n=testnum, replace=False, random_state=128)
            test_flt = sorted(test_flt.index.values)
        self.test_flt = test_flt
        test_srcfiles = self.d_df.loc[test_flt, 'srcfile'].values
        test_tgtfiles = self.d_df.loc[test_flt, 'tgtfile'].values
        test_X = np.array([load_image(path, self.imgsize) for path in test_srcfiles])
        test_y = np.array([load_image(path, self.imgsize) for path in test_tgtfiles])
        doCenteredNorm = getGlobalConfigData(self.d_gcf, self.d_cf, 'centered_normalize_X', 0)
        if doCenteredNorm > 0:
            centered_norm(test_X)
        self.test = DataSet(test_X, test_y, test_srcfiles, test_tgtfiles)

    def run(self):
        device = getGlobalConfigData(self.d_gcf, self.d_cf, '.device', 'cpu')
        dataformat = 'channels_last' if device.lower() == 'cpu' else 'channels_first'
        random_seed = getGlobalConfigData(self.d_gcf, self.d_cf, '.random_seed', 123)
        epoch = getConfigData(self.d_cf, ".epoch", 2)

        self.loadDataSet()

        test_X = self.test.input_images
        self.d_df.loc[self.test_flt, 'applyfile'] = self.d_df.loc[self.test_flt, 'name'].apply(lambda x: str(x)+ '_CSemApply.jpg')
        y_pred_filenames = self.d_df.loc[self.test_flt, 'applyfile'].values
        applypath = self.stagepath
        cnn = ConvNN(random_seed=random_seed,imgsize=self.imgsize, data_format=dataformat)
        inxmlfile = getConfigData(self.d_cf, MXP_XML_TAGS[0])
        inxmlfile = os.path.join(self.jobresultpath, inxmlfile)
        parser = MXPStageXml(inxmlfile, option=MXP_XML_TAGS[0])
        modelpath = getConfigData(parser.geticf(), '.model')
        if not os.path.exists(modelpath):
            raise ValueError("%s, Error, does not exists model path: %s" % (self.stagename, modelpath))
        cnn.load(epoch=epoch, path=modelpath)
        cnn.model_apply(test_X, y_pred_filenames, path=applypath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xml-driving CNN sem job')
    parser.add_argument('jobpath', help='cnn sem job path')
    args = parser.parse_args()
    jobpath = args.jobpath
    if jobpath is None:
        parser.print_help()
        parser.exit()
        # jobpath = './samplejob'
        # print('No jobpath is inputed, use sample job path: %s' % jobpath)
    print(str(vars(args)))
    myjob = MXPJob(jobpath)
    myjob.run()