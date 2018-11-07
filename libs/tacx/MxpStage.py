# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-22 23:17:50

MXP Stage base

Last Modified by:  ouxiaogu
"""

import os, os.path
import xml.etree.ElementTree as ET
from subprocess import call

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
from XmlUtil import dfFromConfigMapList, getConfigData, getGlobalConfigData, dfToMxpOcf, indentCf, setConfigData
from logger import logger
log = logger.getLogger(__name__)

__all__ = ['MXP_XML_TAGS', 'MXP_RESULT_OPTIONS',
         'MxpStage', 'MxpStageXmlParser',]

MXP_XML_TAGS = ["inxml", "outxml"]
MXP_RESULT_OPTIONS = ["occfs", "osumccfs", "osumkpis"]

class MxpStageXmlParser(object):
    """docstring for MxpStageXmlParser"""
    def __init__(self, xmlfile, option=MXP_XML_TAGS[1]):
        super(MxpStageXmlParser, self).__init__()
        self.xmlfile = xmlfile
        self.option = option
        self.__build()

    def __build(self):
        if self.option==MXP_XML_TAGS[0]:
            self.loadicf()
        elif self.option==MXP_XML_TAGS[1]:
            self.loadocf()
            self.loadosumcf()

    def loadicf(self):
        try:
            self.icf = ET.parse(self.xmlfile).getroot().find("result")
            log.debug("icf tag: "+self.icf.tag)
        except AttributeError:
            raise AttributeError("%s has no attribute 'tag'" % self.icf)
        return self.icf

    def loadocf(self):
        try:
            self.ocf = ET.parse(self.xmlfile).getroot().find("result")
            log.debug("ocf tag: "+self.ocf.tag)
        except AttributeError:
            raise AttributeError("%s has no attribute 'tag'" % self.ocf)

    def loadosumcf(self):
        self.osumcf = self.ocf.find("summary")
        try:
            log.debug("osumcf tag: "+self.osumcf.tag)
        except:
            log.debug("No summary tag in input xml： %s" % self.xmlfile)
            pass

    def iccfs2df(self):
        return dfFromConfigMapList(self.icf, ".pattern")

    def occfs2df(self):
        """
        Returns
        -------
        ret : DataFrame object
            all the pattern's results in xmlfile
        """
        return dfFromConfigMapList(self.ocf, ".pattern")

    def osumccfs2df(self):
        """
        Returns
        -------
        ret : DataFrame object
            all the pattern's summary in xmlfile"""
        return dfFromConfigMapList(self.osumcf, ".pattern")

    def osumkpis2df(self):
        """
        Returns
        -------
        ret : DataFrame object\
            all the pattern's KPIsummary in xmlfile"""
        return dfFromConfigMapList(self.osumcf, ".pattern/KPI")

class MxpStage(object):
    """docstring for MxpStage"""
    datarelpath = os.sep.join(['h', 'data', 'dummydb', 'MXP', 'job1']) # r"h/data/dummydb/MXP/job1"
    resultrelpath = os.sep.join(['h', 'cache', 'dummydb', 'result', 'MXP', 'job1']) # r"h/cache/dummydb/result/MXP/job1"

    def __init__(self, gcf, cf, stagename, jobpath):
        self.d_gcf = gcf
        self.d_cf = cf
        self.stagename = stagename
        self.jobpath = jobpath
        jobresultabspath = os.path.join(self.jobpath, self.resultrelpath)
        if not os.path.exists(jobresultabspath):
            log.debug('job result abspath: {}'.format(jobresultabspath))
            os.makedirs(jobresultabspath)
        self.jobresultabspath = jobresultabspath
        self.stageresultrelpath = stagename+'result1'
        stageresultabspath = os.path.join(jobpath, self.resultrelpath, self.stageresultrelpath)
        if not os.path.exists(stageresultabspath):
            log.debug('stage result abspath: {}'.format(stageresultabspath))
            os.makedirs(stageresultabspath)
        self.stageresultabspath = stageresultabspath

        self.__build()

    def __symbollink(self):
        self.dataabspath = os.path.join(self.jobpath, self.datarelpath)
        for item in os.listdir(self.dataabspath):
            try:
                os.symlink(os.path.join(self.dataabspath, item), os.path.join(self.resultrelpath, item))
            except OSError:
                call('ln -s {} {}'.format(os.path.join(self.dataabspath, item), 
                    os.path.join(self.resultrelpath, item)), shell=True)

    def __loadCfg(self):
        inxmlfile = getConfigData(self.d_cf, MXP_XML_TAGS[0])
        inxmlfile_abspath = os.path.join(self.jobresultabspath, inxmlfile)
        icf_Parser = MxpStageXmlParser(inxmlfile_abspath, option=MXP_XML_TAGS[0])
        self.d_df_patterns = icf_Parser.iccfs2df()
        self.d_icf = icf_Parser.icf
        self.d_ocf = self.d_icf # directly use reference, without copy here

    def __build(self):
        self.__symbollink()

        if "init" not in self.stagename.lower():
            self.__loadCfg()

    def save(self, path, viaDf=False, extraNodes=None):
        if viaDf:
            ocf = dfToMxpOcf(self.d_df_patterns)
        else:
            ocf = self.d_ocf
        if extraNodes is not None:
            for key, val in extraNodes.items():
                setConfigData(ocf, key, val)
        root = ET.Element('root')
        root.append(ocf)
        indentCf(root)
        log.debug("Result save path: %s\n" % (path))
        tree = ET.ElementTree(root)
        tree.write(path, encoding="utf-8", xml_declaration=True)
        log.info("Result saved at %s\n" % (path))