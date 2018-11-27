# -*- coding: utf-8 -*-
"""
Created: hshi & peyang, 2018-01-25 12:02:59

TachyonJob: Base Class to hold tachyon job, 
Independent to TACX FEM+ python api now, no need tachyon_python
Since GUI don't provide the base TAC job api, use own JobInfoXml parser instead

    * get/set job status
    * get/set Job option
    * submit job

Last Modified by:  ouxiaogu
"""

import xml.etree.ElementTree as ET
import time
import subprocess

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
from XmlUtil import JobInfoXml, getConfigData
from logger import logger
from StrUtil import parseKW, buildKW

log = logger.getLogger(__name__)

class Job(object):
    NEWSTATUS = frozenset(['New'])
    TERMINATESTATUS = frozenset(['Aborted', 'Cancelled', 'Failed', 'Exited', 'Terminated'])
    FINISHSTATUS = frozenset(['Done'])
    RUNINGSTATUS = frozenset(['Preprocessing', 'Preproc Done', 'Running', 'OnHold'])

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
            raise IOError("Job xml not exists at: {}".format(jobxml))
        self.jobxml = jobxml

    def submit(self):
        release_xml = os.path.join(self.jobpath, 'jobinfo', 'release.xml')
        log.debug('Job submit:')
        try:
            root = ET.parse(release_xml).getroot()
            relroot = JobInfoXml(root.find('programs'))
            appdir = relroot.getConfigData(".item/[jobtype='release']", tag='program')
            
            tperl = os.path.join(appdir, '../../bin/tachyon_perl')
            submitjob_pl = os.path.join(appdir, '../../perl/submitjob.pl')

            command = [tperl, submitjob_pl, '-d', self.jobpath]
            log.debug(' '.join(command))
            subprocess.call(command)
        except:
            log.error('failed to parse release application directory from {}'.format(release_xml))
            sys.exit(-1)

    @property
    def status(self):
        root = ET.parse(self.jobxml).getroot()
        jobroot = JobInfoXml(root)
        try:
            status = jobroot.getConfigData(".item/[name='status']")
        except KeyError:
            return 'New'
        return status

    def getStatus(self):
        return self.status

    def waitTillDone(self):
        while True:
            status = self.status
            if status in self.FINISHSTATUS:
                break
            elif status in self.TERMINATESTATUS:
                raise RuntimeError('Job {}!'.format(status))
            else:
                time.sleep(1)

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
        options = buildKW(dict(kwargs))
        self.setJobInfoConfigData('options', options)

    def setJobInfoConfigData(self, key, val):
        tree = ET.parse(self.jobxml) # for xml writing
        root = tree.getroot()
        jobroot = JobInfoXml(root)
        jobroot.setConfigData("item/[name='{}']".format(key), val)
        tree.write(self.jobxml)

if __name__ == '__main__':
    jobpath = r'/gpfs/DEV/FEM/peyang/release/E8.0/MOD9944/job2_2D_CD+EP_align_correction_GN_clone'
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