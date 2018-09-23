# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-22 23:17:50

MXP Stage base

Last Modified by: ouxiaogu
"""

import os, os.path
import xml.etree.ElementTree as ET
from subprocess import call

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
from XmlUtil import dfFromConfigMapList, getConfigData, getGlobalConfigData, dfToMxpOcf
import logger
log = logger.setup("MxpStage")

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
    datarelpath = r"h/data/dummydb/MXP/job1"
    resultrelpath = r"h/cache/dummydb/result/MXP/job1"

    def __init__(self, gcf, cf, stagename, jobpath):
        self.d_gcf = gcf
        self.d_cf = cf
        self.stagepath = stagepath
        self.stagename = stagename
        self.jobpath = jobpath
        jobresultabspath = os.path.join(jobpath, resultrelpath, stagename+'result1')
        if not os.path.exists(jobresultabspath):
            os.mkdirs(jobresultabspath)
        self.jobresultabspath = jobresultabspath

        self.__build()

    def __symbollink(self):
        self.dataabspath = os.path.join(self.jobpath, self.datarelpath)
        self.resultabspath = os.path.join(self.jobpath, self.resultrelpath)
        for item in os.listdir(self.dataabspath):
            try:
                os.symlink(os.path.join(self.dataabspath, item), os.path.join(self.resultrelpath, item))
            except OSError:
                call('ln -s {} {}'.format(os.path.join(self.dataabspath, item), 
                    os.path.join(self.resultrelpath, item)), shell=True)

    def __readDf(self):
        inxmlfile = getConfigData(self.d_cf, MXP_XML_TAGS[0])
        inxmlfile_abspath = os.path.join(self.resultrelpath, inxmlfile)
        icf_Parser = MxpStageXmlParser(inxmlfile_abspath, option=MXP_XML_TAGS[0])
        self.d_df = icf_Parser.geticcfs()

    def __build(self):
        self.__symbollink()

        if "init" not in self.stagename.lower():
            self.__readDf()

    def save(self, path):
        ocf = dfToMxpOcf(self.d_df)
        log.debug("Result save path: %s\n" % (path))
        tree = ET.ElementTree(ocf)
        tree.write(path, encoding="utf-8", xml_declaration=True)
        log.debug("Result saved at %s\n" % (path))