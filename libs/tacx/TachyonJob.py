"""
-*- coding: utf-8 -*-
Created: hshi & peyang, 2018-01-25 12:02:59

Last Modified by: ouxiaogu

TachyonJob: Base Class to hold tachyon job
Since GUI don't provide the base TAC job api, use own xml parser instead

    * get/set job status
    * get/set Job option

"""

import os.path
import xml.etree.ElementTree as ET
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
from XmlUtil import JobInfoXml
import logger
from StrUtil import parseKW, buildKW

log = logger.setup('TachyonJob', 'info')

NEWSTATUS = frozenset(['New'])
TERMINATESTATUS = frozenset(['Aborted', 'Cancelled', 'Failed', 'Exited', 'Terminated'])
FINISHSTATUS = frozenset(['Done'])
RUNINGSTATUS = frozenset(['Preprocessing', 'Preproc Done', 'Running', 'OnHold'])

class Job(object):
    """docstring for Job"""
    def __init__(self, jobpath):
        super(Job, self).__init__()
        self.__buildjob(jobpath)

    def __buildjob(self, jobpath):
        if not os.path.exists(jobpath):
            e = "Job not exists at: {}".format(jobpath)
            raise IOError(e)
        jobpath = os.path.abspath(jobpath)
        self.jobpath = jobpath

        jobxml = os.path.join(self.jobpath, 'jobinfo', 'job.xml')
        if not os.path.exists(jobxml):
            sys.exit("Job xml not exists at: {}".format(jobxml))
        self.jobxml = jobxml

    @property
    def status(self):
        root = ET.parse(self.jobxml).getroot()
        try:
            status = getConfigData(root, ".item/[name='status']")
            return status
        except KeyError:
            return 'New'

    def getStatus(self):
        return self.status

    def updateStatus(self, stagename='', message='', curstage=1, totstage=1, jobstart=False, jobdone=False, jobabort=False, descr=None):
        if curstage > totstage:
            raise ValueError("Input current stage %d larger than total stage number %d" % (curstage, totstage))

        tree = ET.parse(self.jobxml) # for xml writing
        root = tree.getroot()
        jobroot = JobInfoXml(root)
        s_time = time.strftime('%Y-%m-%d %H:%M:%S')
        if jobstart or curstage==0:
            try:
                if descr is not None:
                    log.debug('descr=%s' % descr)
                    jobroot.setConfigData("item/[name='descr']", descr)
                jobroot.setConfigData("item/[name='status']", 'Submitted')
                jobroot.setConfigData("item/[name='message']", "%s[%d/%d]:%s" % (stagename, curstage+1, totstage, 'running'))
                jobroot.setConfigData("item/[name='submittime']", )
                jobroot.setConfigData("item/[name='starttime']", s_time)
                jobroot.setConfigData("item/[name='endtime']", '')
            except AttributeError:
                jobroot.addChildNode("message", '%s[%d/%d]:%s' % (stagename, curstage, totstage, 'running'))
                jobroot.addChildNode("submittime", s_time)
                jobroot.addChildNode("starttime", s_time)
                jobroot.addChildNode("endtime", '')
                jobroot.addChildNode("status", 'Submitted')
        elif jobabort:
            jobroot.setConfigData("item/[name='message']", '%s[%d/%d]:%s(%s)' % (stagename, curstage, totstage, 'Aborted', message))
            jobroot.setConfigData("item/[name='endtime']", s_time)
            jobroot.setConfigData("item/[name='status']", 'Aborted')
        elif jobdone or curstage==totstage:
            jobroot.setConfigData("item/[name='message']", '%s[%d/%d]:%s(%s)' % (stagename, curstage, totstage, 'Done', message))
            jobroot.setConfigData("item/[name='endtime']", s_time)
            jobroot.setConfigData("item/[name='status']", 'Done')
        else:
            jobroot.setConfigData("item/[name='message']", '%s[%d/%d]:%s(%s)' % (stagename, curstage, totstage, 'running', message))
            jobroot.setConfigData("item/[name='status']", 'Running')
        tree.write(self.jobxml)

    def getOption(self, astype="dict"):
        root = ET.parse(self.jobxml).getroot()
        jobroot = JobInfoXml(root)
        try:
            options = jobroot.getConfigData(".item/[name='options']")
            if astype == "dict":
                options = parseKW(options)
            return options
        except KeyError:
            return None

    def setOption(self, **kwargs):
        tree = ET.parse(self.jobxml) # for xml writing
        root = tree.getroot()
        jobroot = JobInfoXml(root)
        options = buildKW(dict(kwargs));
        jobroot.setConfigData(".item/[name='options']", options)
        tree.write(self.jobxml)

if __name__ == '__main__':
    jobpath = r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/Calaveras_v2/peyang/jobs/8GF02/06_study_c2c_id2db_Case2E_IMEC_EP5_old_bin'
    from FileUtil import gpfs2WinPath
    jobpath = gpfs2WinPath(jobpath)
    print(jobpath)
    m_job = Job(jobpath)
    print (m_job.status)
    joboptions = m_job.getOption()
    print(joboptions)
    print(buildKW(joboptions))

    joboptions = "enable=1-2000,MXP.debug=1,avx=1"
    # m_job.setOption(a=1, b=2)
    m_job.setOption(**parseKW(joboptions))