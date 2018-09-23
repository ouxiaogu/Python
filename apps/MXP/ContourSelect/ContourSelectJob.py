# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-23 11:35:37

ContourSelect job

Last Modified by: ouxiaogu
"""

import sys
import os.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../../../libs/tacx")
from MxpJob import MxpJob
from MxpStage import MXP_XML_TAGS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../../../libs/common")
import logger
log = logger.setup("ContourSelectJob")

###############################################################################
# MXP Contour Select Job Stage Register Area
from ContourLabeling import ContourSelLabelStage
from ContourSelModelCal import ContourSelCalStage
from ContourSelModelApply import ContourSelApplyStage

STAGE_REGISTER_TABLE = {
    'ContourSelectDataLabeling': 'ContourSelLabelStage',
    'ContourSelectModelCalibration': 'ContourSelCalStage',
    'ContourSelectModelApply': 'ContourSelApplyStage'
}
###############################################################################

class ContourSelJob(MxpJob):
    """
    ContourSelJob: ContourSelect job
    """
    def __init__(self, jobpath)
        super(ContourSelJob, self).__init__(jobpath)
    
    def run(self, range):
        allstages = self.getAllMxpStages()
        gcf = self.mxproot.find('.global')
        for stage in allstages:
            stagename, enablenum = stage
            log.info("Stage %s%d starts\n" % (stagename, enablenum))
            cf = self.getStageConfig(stage)
            stagepath = self.resultAbsPath('{}{}'.format(stagename, enablenum))
            curstage = eval(STAGE_REGISTER_TABLE[stagename])(gcf, cf, stagename, stagepath)
            curstage.run()
            outxmlfile = self.getStageIOFile(stage, option=MXP_XML_TAGS[1])
            curstage.save(outxmlfile)
            log.info("Stage %s%d successfully finished\n" % (stagename, enablenum))

def main():
    parser = argparse.ArgumentParser(description='MXP Contour Selection job in python')
    parser.add_argument('jobpath', help='MXP Contour Selection job path')
    args = parser.parse_args()
    jobpath = args.jobpath
    if jobpath is None:
        parser.print_help()
        parser.exit()

    print(str(vars(args)))
    myjob = ContourSelJob(jobpath)
    myjob.run()

if __name__ == '__main__':
    main()