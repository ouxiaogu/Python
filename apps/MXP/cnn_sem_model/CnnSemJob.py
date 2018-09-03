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
import logger
log = logger.setup("MXPJob", "debug")

MXP_XML_TAGS = ["inxml", "outxml"]
MXP_RESULT_OPTIONS = ["occfs", "osumccfs", "osumkpis"]
PATTERN_DEFAULT_ATTR =\
'''name costwt    usage   srcfile    tgtfile  imgpixel   offset_x    offset_y
1   1 training   1_se.bmp    1_image.bmp    1   0   0'''
STAGE_REGISTER_TABLE = {
    'init': 'InitStage',
    'DLSEMCalibration': 'CSemCalStage',
    'DLSEMCheck': 'CSemChkStage'
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
        for stage in allstages:
            stagename, enablenum = stage
            log.info("Stage %s%d starts\n" % (stagename, enablenum))
            df = None
            if stagename != "init":
                inxmlfile = self.getStageIOFile(stage, option="inxml")
                stagexml = MXPStageXml(inxmlfile)
                df = stagexml.geticcfs()
            cfg = self.getStageConfig(stage)
            stagepath = self.resultAbsPath(''.join(stage)+'result')
            curstage = eval(STAGE_REGISTER_TABLE[stagename])(cfg, df, stagepath)
            curstage.run()
            outxmlfile = self.getStageIOFile(stage, option="outxml")
            curstage.save(outxmlfile)
            log.info("Stage %s%d successfully finished\n" % (stagename, enablenum))

class MXPStageXml(object):
    """docstring for MXPStageXml"""
    def __init__(self, xmlfile):
        super(MXPStageXml, self).__init__()
        self.xmlfile = xmlfile
        self.buildschema()

    def buildschema(self):
        self.geticf()
        self.getocf()
        self.getosumcf()

    def geticf(self):
        try:
            self.icf = ET.parse(self.xmlfile).getroot().find("result")
            log.debug("ocf tag: "+self.ocf.tag)
        except AttributeError:
            raise AttributeError("%s has no attribute 'tag'" % self.ocf)
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
            log.debug("No summary tag in input xml： %s" % self.xml)
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
    def __init__(self, cf, df, stagepath):
        self.d_cf = cf
        self.d_df = df
        if not os.path.exists(stagepath):
            os.mkdir(stagepath)
        self.d_path = stagepath

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
            raise IOError("data_dir not exists at: {}\n".format(dataDir))
        log.info('Going to glob dataset\n')
        log.info('Now going to read path for both input and target images\n')
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
        self.d_df = df

        if doshuffle > 0:
            self.d_df = self.d_df.sample(frac=1).reset_index(drop=True) # shuffle

        divides = [min(1, int(d/100.*nsamples)) for d in divide_rule]
        divides[-1] = nsamples - sum(divides[:-1])
        assert(all(divides>=0))
        divides = np.cumsum(divides)
        def usagefunc(x):
            usage = 'training'
            if x.index > divides[0] and x.index < divides[1]:
                usage = 'validation'
            elif x.index > divides[1] and x.index < divides[2]:
                usage = 'test'
            return usage
        self.d_df[:, 'usage'] = self.d_df['usage'].apply(usagefunc(x), axis=1)

class CSemCalStage(MxpStage):
    def loadDataSet(self):
        from dataset import DataSet, load_image
        nsamples = len(self.d_df)
        imgsize = getConfigData(self.d_cf, ".imgsize", 256)
        normalize_X = getConfigData(self.d_cf, ".normalize_X", 1)
        input_images_file_name = self.d_df.ix[:, 'srcfile'].tolist()
        target_images_file_name = self.d_df.ix[:, 'tgtfile'].tolist()
        input_images = np.array([load_image(path, imgsize) for path in input_images_file_name])
        target_images = np.array([load_image(path, imgsize) for path in target_images_file_name])
        if normalize_X > 0:
            means = [np.mean(input_images[i], dtype=np.float64) for i in range(nsamples)]
            stds = [np.std(input_images[i], dtype=np.float64) for i in range(nsamples)]
            input_images = np.array([(input_images[i] - means[i])/stds[i] if stds[i]!=0 else np.zeros_like(input_images[i])  for i in range(nsamples)])

        train_indice = (self.d_df.usage=='training').nonzero()[0]
        train_input_images = input_images[train_indice]
        train_target_images = target_images[train_indice]
        train_input_images_file_name = input_images_file_name[train_indice]
        train_target_images_file_name = target_images_file_name[train_indice]

        validation_indice = (self.d_df.usage=='validation').nonzero()[0]
        validation_input_images = input_images[validation_indice]
        validation_target_images = target_images[validation_indice]
        validation_input_images_file_name = input_images_file_name[validation_indice]
        validation_target_images_file_name = target_images_file_name[validation_indice]

        self.train = DataSet(train_input_images, train_target_images, train_input_images_file_name, train_target_images_file_name)
        self.valid = DataSet(validation_input_images, validation_target_images, validation_input_images_file_name, validation_target_images_file_name)

        log.info("Complete reading input data")
        log.info("Number of files in Training-set:\t\t{}".format(len(train_indice)))
        log.info("Number of files in Validation-set:\t{}".format(len(validation_indice)))

    def run(self):
        from convNN import ConvNN
        self.loadDataSet()
        learning_rate = getConfigData(self.d_cf, ".learning_rate", 1e-4)
        X_train, y_train = self.train.input_images, self.train.target_images
        X_valid, y_valid = self.valid.input_images, self.valid.target_images
        log.info('Training:\t{}\t{}'.format(str(X_train.shape), str(y_train.shape)))
        log.info('Validation:\t{}\t{}'.format(str(X_valid.shape), str(y_valid.shape)))
        cnn = ConvNN(batchsize=8, random_seed=123, imgsize=_imgsize)
        cnn.train(training_set=(X_train, y_train),
                  validation_set=(X_valid, y_valid))
        self.modelpath=self.stagepath+'/tflayers-model/'
        cnn.save(epoch=20, path=self.modelpath)

    def save(self, path):
        xml = ['<root>']
        xml.append('\t<result>')
        xml.append('\t\t<model>{}</model>'.format(self.modelpath))
        xml.extend(df.apply(dfRowToXmlStream, axis=1))
        xml.append('\t</result>')
        xml.append('</root>')
        ss = '\n'.join(xml)
        log.debug("Result save path: %s\n" % (path))
        with open(path, 'w') as fout:
            fout.write(ss)
        log.info("Result saved at %s\n" % (path))

class CSemChkStage(MxpStage):
    def loadDataSet(self):
        from dataset import DataSet, load_image
        

if __name__ == '__main__':
    nodestr = """<pattern>
    <kpi>0.741096070383657</kpi>
    <test>
        <key>name</key>
        <value>213.</value>
        <options><enable>1-2000</enable></options>
    </test>
    <name>13</name>
</pattern>"""
    root = ET.fromstring(nodestr)
    kpi = root.find(".kpi")
    print(kpi.tag, len(kpi))
    test = root.find(".test")
    print (test.tag, len(test))
    print (root.find("./test/options/enable").text)

    print (root.tag)
    print (getConfigMap(root))
    print (getRecurConfigMap(root))
