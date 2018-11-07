"""
-*- coding: utf-8 -*-
Created: peyang, 2018-01-26 13:20:55

Last Modified by:  ouxiaogu

MxpJob: Class to hold tachyon MXP job, derived from class TachyonJob
"""
from TachyonJob import Job
from collections import OrderedDict
import xml.etree.ElementTree as ET
import re
import argparse

import sys
import os.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
from XmlUtil import getSonConfigMap, getElemText
from StrUtil import parseKW
from logger import logger
log = logger.getLogger(__name__)

###############################################################################
# MXP Job Stage Register Area
## MXP base stage
from MxpStage import *

STAGE_REGISTER_TABLE = {

}
###############################################################################

__all__ = ['MxpJob']

class MxpJob(Job):
    """docstring for MxpJob"""
    datarelpath = r"h/data/dummydb/MXP/job1"
    resultrelpath = r"h/cache/dummydb/result/MXP/job1"
    
    def __init__(self, jobpath):
        super(MxpJob, self).__init__(jobpath)
        self.__buildjob()

    def __buildjob(self):
        self.resultabspath = os.path.join(self.jobpath, self.resultrelpath)
        if not os.path.exists(self.resultabspath):
            os.makedirs(self.resultabspath)
        self.mxpxml = os.path.join(self.jobpath, self.datarelpath, 'mxp.xml')

        mxproot = None
        try:
            mxproot = ET.parse(self.mxpxml).getroot().find(".MXP")
        except:
            raise KeyError("MXP tag not find in xml: %s" % self.mxpxml)
        self.mxproot = mxproot

        self.getmxpCfgMap()

    def resultAbsPath(self, basename):
        return os.path.join(self.jobpath, self.resultrelpath, basename)

    def dataAbsPath(self, basename):
        return os.path.join(self.jobpath, self.datarelpath, basename)

    def getEnableRangeList(self):
        """
        Enable range list in MXP job option

        Example
        -------
        Follow the MXP convention
        input: enable = '100-300 + 410 + 490-700'
        output: [(100, 300), (410, 410), (490, 700)]
        explain output: list of len=3, each item is a tuple of (start, end)
            [-1, -1]: means all the stages are enabled
            [min, max]:stage between [min max] are enabled
        """
        range_list = []
        default_range = (-1, -1)
        try:
            options = self.getOption()
            range_ = options["enable"]
            # range_ = self.mxproot.find('./global/options/enable').text
            range_ = parseKW(range_, '+', '-')
            range_list = [tuple(map(int, (k, v))) for k, v in range_.items()]
        except:
            range_list.append(default_range)
        return range_list

    def getmxpCfgMap(self):
        """OrderedDict {(stagename, enable): dict }"""
        mxpCfgMap = OrderedDict()
        for stagecf in list(self.mxproot):
            stagename = stagecf.tag
            enable = ""
            try:
                enable = getElemText(stagecf.find(".enable"))
            except:
                continue
            cfgMap = getSonConfigMap(stagecf)
            mxpCfgMap[(stagename, enable)] = cfgMap
        self.mxpCfgMap = mxpCfgMap
        return mxpCfgMap

    def getAllMxpStages(self, enabled_only=True):
        """
        Returns
        -------
        stages : sorted list by enable number
            items like [(stagename, enable), ...], ascending order
        enabled_only : bool
            if True, just get the enabled_only stages; if False, get all

        Example
        -------
            [('D2DBAlignment', 500), ('ResistModelCheck', 510),
             ('ImageD2DBAlignment', 600), ('ResistModelCheck', 610)]
        """
        range_list = self.getEnableRangeList()
        if (-1, -1) in range_list:
            enabled_only = False
        log.debug('enable range list: {}'.format(range_list))
        inRange = lambda x: any([True if x >= range_[0] and x <= range_[1] else False for range_ in range_list]) 
        stages = []
        for key, stagectx in self.mxpCfgMap.items():
            stagename, enable_ = key
            if type(enable_) == str:
                continue
            if enabled_only and not inRange(enable_):
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
        """
        get the relative path of all stage xml files for both inxml and outxml, 
        by default only export IO files from the enabled stages
        
        Example
        -------
            {('D2DBAlignment', 500): {'outxml': "d2dbalignment500out.xml", ...}, ...}
        """
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

    def getStageID(self, stage=None, stagename=None, enable=None):
        '''
        get unique stage ID, by combining stage name and enable number
        '''
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
        return stage_

    def getStageConfig(self, stage=None, stagename=None, enable=None):
        stage_ = self.getStageID(stage=stage, stagename=stagename, enable=enable)
        if not isinstance(stage_, tuple) and len(stage_) !=2:
            raise ValueError("Wrong input: %s, please input MXP stage name together with enable number, e.g., (Average, 300)\n" % str(stage_))
        stagename_, enable_ = stage_
        for item in list(self.mxproot):
            stagename = item.tag
            enable = ""
            try:
                enable = getElemText(item.find(".enable"))
            except:
                pass
            if stagename == stagename_ and enable==enable_:
                return item
        return None

    def getStageIOFile(self, stage=None, stagename=None, enable=None, option="outxml"):
        """
        get the relative path of stage xml file, default is outxml

        Example
        -------
        The 4 operations below are equivalent 

        1. input stagename='D2DBAlignment', enable=500, option="outxml", output: "d2dbalignment500out.xml"
        2. input enable=500, option="outxml", output: "d2dbalignment500out.xml"
        3. input stage="D2DBAlignment500", option="outxml", output: "d2dbalignment500out.xml"
        4. input stage=("D2DBAlignment", 500), option="outxml", output: "d2dbalignment500out.xml"
        """
        stage_ = self.getStageID(stage=stage, stagename=stagename, enable=enable)
        
        stageIOFiles = self.getAllStageIOFiles() if not hasattr(self, 'stageIOFiles') else self.stageIOFiles
        if stage_ not in stageIOFiles:
            raise KeyError("Input stage %s not in: %s" % (stage_, str(stageIOFiles.keys())))
        if option not in MXP_XML_TAGS:
            raise KeyError("Input option %s not in: %s" % (option, str(MXP_XML_TAGS)))
        xmlfile = stageIOFiles[stage_][option]
        # xmlfile = self.dataAbsPath(xmlfile) if option == MXP_XML_TAGS[0] else self.resultAbsPath(xmlfile)
        xmlfile = self.resultAbsPath(xmlfile) # init stage without inxml, other stages inxml and outxml are all in result folder
        return xmlfile

    def getAllStageOut(self):
        """
        occf results of all the stage, i.e., outxml result config
        
        Returns
        -------
        dfs : OrderedDict
            key is tuple like, value is dataframe object

        Example
        -------
        key=("D2DBAlignment", 500), value: df parsed from the result/pattern of outxml "d2dbalignment500out.xml"
        """
        stageIOFiles = self.getAllStageIOFiles()
        dfs = OrderedDict()
        for stagename in stageIOFiles.keys():
            dfs[stagename] = self.getStageOut(self, stagename)
        return dfs

    def getStageOut(self, stage=None, stagename=None, enable=None):
        """Refer to MxpStageXmlParser.occfs2df"""
        return self.getStageResultFactory(stage=stage, stagename=stagename, enable=enable, result_option="occfs")

    def getStageSummary(self, stage=None, stagename=None, enable=None):
        """Refer to MxpStageXmlParser.osumccfs2df"""
        return self.getStageResultFactory(stage=stage, stagename=stagename, enable=enable, result_option="osumccfs")

    def getStageSummaryKpi(self, stage=None, stagename=None, enable=None):
        """Refer to MxpStageXmlParser.osumkpis2df"""
        return self.getStageResultFactory(stage=stage, stagename=stagename, enable=enable, result_option="osumkpis")

    def getStageResultFactory(self, stage=None, stagename=None, enable=None, result_option="occfs"):
        xmlfile = self.getStageIOFile(stage=stage, stagename=stagename, enable=enable)
        xmlfile = self.resultAbsPath(xmlfile)
        parser = MxpStageXmlParser(xmlfile)
        if result_option is None:
            result_option = MXP_RESULT_OPTIONS[0]
            
        if result_option == MXP_RESULT_OPTIONS[0]:
            return parser.occfs2df()
        elif result_option == MXP_RESULT_OPTIONS[1]:
            return parser.osumccfs2df()
        elif result_option == MXP_RESULT_OPTIONS[2]:
            return parser.osumkpis2df()
        else:
            raise KeyError("Input MXP result option %s not in: %s" % (result_option, str(MXP_RESULT_OPTIONS)))

    def run(self):
        allstages = self.getAllMxpStages(enabled_only=True)
        gcf = self.mxproot.find('.global')
        for stage in allstages:
            stagename, enablenum = stage
            log.info("Stage %s%d starts\n" % (stagename, enablenum))
            cf = self.getStageConfig(stage)
            stagestr = '{}{}'.format(stagename, enablenum)
            curstage = eval(STAGE_REGISTER_TABLE[stagename])(gcf, cf, stagestr, self.jobpath) # MxpStage
            curstage.run()
            outxmlfile = self.getStageIOFile(stage, option=MXP_XML_TAGS[1])
            curstage.save(outxmlfile)
            log.info("Stage %s%d successfully finished\n" % (stagename, enablenum))

def test_mxpjob():
    jobpath = r'C:\Localdata\D\Note\Python\apps\MXP\ContourSelect\samplejob'
    from FileUtil import gpfs2WinPath
    jobpath = gpfs2WinPath(jobpath)
    m_job = MxpJob(jobpath)
    m_stages =  m_job.getAllMxpStages(enabled_only=False)
    print(m_stages)
    print(m_job.getEnableRangeList())
    print (m_job.getStageOut("ContourExtraction400"))

def main():
    parser = argparse.ArgumentParser(description='xml-driving MXP job')
    parser.add_argument('jobpath', help='MXP job path')
    args = parser.parse_args()
    jobpath = args.jobpath
    if jobpath is None:
        parser.print_help()
        parser.exit()

    print(str(vars(args)))
    myjob = MxpJob(jobpath)
    myjob.run()

if __name__ == '__main__':
    # test_mxpjob()

    main()