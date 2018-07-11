"""
-*- coding: utf-8 -*-
Created: peyang, 2018-01-26 13:20:55

Last Modified by: peyang

MXPJob: Class to hold tachyon MXP job, derived from class TachyonJob
"""
from TachyonJob import Job
import logger
from collections import OrderedDict
import os.path
import xml.etree.ElementTree as ET
from XmlUtil import getRecurConfigMap, getElemText, dfFromConfigMapList
import re

logger.initlogging(debug=False)
log = logger.getLogger("MXPJob")

MXP_XML_FILE_TAGS = ["inxml", "outxml"]
MXP_RESULT_OPTIONS = ["occfs", "osumccfs", "osumkpis"]

class MXPJob(Job):
    """docstring for MXPJob"""
    def __init__(self, jobpath):
        super(MXPJob, self).__init__(jobpath)
        self.__buildjob()

    def __buildjob(self):
        mxpxmls = [r"h/data/dummydb/smo/job1/smo.xml", r"h/data/dummydb/MXP/job1/mxp.xml"]
        self.resultrelpath = r"h/cache/dummydb/result/MXP/job1/"

        mxpxml_ = ''
        existed = False
        for xml_ in mxpxmls:
            mxpxml_ = os.path.join(self.jobpath, xml_)
            if os.path.exists(mxpxml_):
                log.debug("MXP xml exists at %s" % mxpxml_)
                existed = True
                break
        if not existed:
            raise IOError("MXP xml doesn't exist for job: %s" % self.jobpath)
        self.mxpxml = mxpxml_
        self.getmxpCfgMap()

    def resultAbsPath(self, basename):
        return os.path.join(self.jobpath, self.resultrelpath, basename)

    def getEnableRange(self):
        """
        Enable range in MXP job option

        Return:
            range_:
                [-1, -1]: means all the stages are enabled
                [min, max]:stage between [min max] are enabled
        """
        range_ = [-1, -1]
        try:
            options = self.getOption()
            range_ = options["enable"]
            range_ = list(map(int, range_.split('-')))
        except:
            pass
        return range_

    def getmxpCfgMap(self):
        """OrderedDict {(stagename, enable): dict }"""
        mxproot = None
        try:
            mxproot = ET.parse(self.mxpxml).getroot().find(".MXP")
        except KeyError:
            raise KeyError("MXP tag not find int xml: %s" % self.mxpxml)
        mxpCfgMap = OrderedDict()
        for item in list(mxproot):
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

    def getAllMxpStages(self, enabled_only=True):
        """
        Return:
            Stages as sorted [(stagename, enable)] by enable

        Example:
            [('D2DBAlignment', 500), ('ResistModelCheck', 510),
             ('ImageD2DBAlignment', 600), ('ResistModelCheck', 610)]
        """
        range_ = self.getEnableRange()
        if set(range_) == set([-1, -1]):
            enabled_only = False
        inRange = lambda x: True if x >= range_[0] and x <= range_[1] else False
        stages = []
        for key, stagectx in self.mxpCfgMap.items():
            stagename, enable_ = key
            if enable_ == "":
                continue
            if enabled_only:
                if inRange(enable_):
                    stages.append((stagename, enable_))
            else:
                stages[enable_] = stagename

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
        return stages

    def getAllStageXmlFiles(self):
        """get the relative path of all stage xml files
        For both inxml and outxml"""
        stages = self.getAllMxpStages()
        stagecfs = OrderedDict()
        for key in stages:
            if not stagecfs.has_key(key):
                stagecfs[key] = OrderedDict()
            for tag in MXP_XML_FILE_TAGS:
                try:
                    stagecfs[key][tag] = self.mxpCfgMap[key][tag]
                except KeyError:
                    pass
        self.stagecfs = stagecfs
        return stagecfs

    def getStageXmlFile(self, stage_, stagename=None, enable=None, option="outxml"):
        """get the relative path of stage xml file, default is outxml"""
        if stagename is not None and enable is not None:
            stage_ = (stagename, enable)
        else:
            try:
                if not isinstance(stage_, str):
                    raise ValueError
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
                else:
                    raise ValueError
            except ValueError:
                raise ValueError("Wrong input: %s, please input MXP stage name together with enable number, e.g., Average300" % str(stage_))

        stagecfs = self.getAllStageXmlFiles()
        if not stagecfs.has_key(stage_):
            raise KeyError("Input stage %s not in: %s" % (stage_, str(stagecfs.keys())))
        if option not in MXP_XML_FILE_TAGS:
            raise KeyError("Input option %s not in: %s" % (option, str(MXP_XML_FILE_TAGS)))
        return stagecfs[stage_][option]

    def getStageOut(self, stage):
        """Refer to MXPOutXml.getoccfs"""
        return self.getStageResultFactory(stage, "occfs")

    def getAllStageOut(self):
        """occf results of all the stage"""
        stagecfs = self.getAllStageXmlFiles()
        dfs = OrderedDict()
        for stagename in stagecfs.keys():
            dfs[stagename] = self.getStageOut(self, stagename)
        return dfs

    def getStageSummary(self, stage):
        """Refer to MXPOutXml.getosumccfs"""
        return self.getStageResultFactory(stage, "osumccfs")

    def getStageSummaryKpi(self, stage):
        """Refer to MXPOutXml.getosumkpis"""
        return self.getStageResultFactory(stage, "osumkpis")

    def getStageResultFactory(self, stage, option="occfs"):
        xmlfile = self.getStageXmlFile(stage)
        xmlfile = self.resultAbsPath(xmlfile)
        parser = MXPOutXml(xmlfile)

        if option == "occfs":
            return parser.getoccfs()
        elif option == "osumccfs":
            return parser.getosumccfs()
        elif option == "osumkpis":
            return parser.getosumkpis()
        else:
            raise KeyError("Input MXP result option %s not in: %s" % (option, str(MXP_RESULT_OPTIONS)))

class MXPOutXml(object):
    """docstring for MXPOutXml"""
    def __init__(self, xmlfile):
        super(MXPOutXml, self).__init__()
        self.xmlfile = xmlfile
        self.buildschema()

    def buildschema(self):
        self.getocf()
        self.getosumcf()

    def getocf(self):
        self.ocf = ET.parse(self.xmlfile).getroot().find("result")
        try:
            log.debug("ocf tag: "+self.ocf.tag)
        except AttributeError:
            raise AttributeError("%s has no attribute 'tag'" % self.ocf)

    def getosumcf(self):
        self.osumcf = self.ocf.find("summary")
        log.debug("osumcf tag: "+self.osumcf.tag)

    def getoccfs(self):
        """
        Return:
            DataFrame object, all the pattern's results in xmlfile
        Example:
        """
        return dfFromConfigMapList(self.ocf, ".pattern")

    def getosumccfs(self):
        """
        Return:
            DataFrame object, all the pattern's summary in xmlfile"""
        return dfFromConfigMapList(self.osumcf, ".pattern")

    def getosumkpis(self):
        """
        Return:
            DataFrame object, all the pattern's KPIsummary in xmlfile"""
        return dfFromConfigMapList(self.osumcf, ".pattern/KPI")

if __name__ == '__main__':
    jobpath = r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/Calaveras_v2/peyang/jobs/8GF02/Case3E_GF_EP5_study_c2c_id2db_v1'
    from FileUtil import gpfs2WinPath
    jobpath = gpfs2WinPath(jobpath)
    m_job = MXPJob(jobpath)
    m_stages =  m_job.getAllMxpStages()
    print m_stages
    print m_job.getEnableRange()

    print m_job.getAllStageXmlFiles()
    print m_job.getStageOut("Average300")
    print m_job.getStageSummary("ImageD2DBAlignment600")
    print m_job.getStageSummaryKpi("ResistModelCheck610")