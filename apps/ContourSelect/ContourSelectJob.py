# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-23 11:35:37

ContourSelect job

Last Modified by:  ouxiaogu
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../../libs/tacx")
from MxpJob import MxpJob
from MxpStage import MXP_XML_TAGS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../../libs/common")
from logger import logger
log = logger.getLogger(__name__)

###############################################################################
# MXP Contour Select Job Stage Register Area
from ContourLabeling import ContourSelLabelStage
from ContourSelModelCal import ContourSelCalStage
#from ContourSelModelApply import ContourSelApplyStage

STAGE_REGISTER_TABLE = {
    'ContourSelectDataLabeling': 'ContourSelLabelStage',
    'ContourSelectModelCalibration': 'ContourSelCalStage',
    'ContourSelection': 'ContourSelStage'
}
###############################################################################

class ContourSelJob(MxpJob):
    """
    ContourSelJob: ContourSelect job
    """
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