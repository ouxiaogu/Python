# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-11-03 13:42:29

Contour Select Job Runner

Last Modified by:  ouxiaogu
"""
import argparse

from ContourSelectJob import ContourSelJob

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../../libs/common")
from logger import logger

def main():

    parser = argparse.ArgumentParser(description='MXP Contour Selection job in python')
    parser.add_argument('jobpath', help='job path')
    #parser.add_argument('--enable', help='job enable range, only run stages in range')
    try:
        args = parser.parse_args()
        jobpath = args.jobpath
    except:
        parser.print_help()
        # parser.exit()

        #args = parser.parse_args(['./samplejob1'])
        #args = parser.parse_args(['/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/IMEC/Case02_calaveras_v3/3Tmp/CT_KPI_test/Calaveras_v3_regular_CT_KPI_003_slope_modified_revert_all_patterns'])
        args = parser.parse_args(['/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/IMEC/Case02_calaveras_v3/3Tmp/ContourSelection/020_AEI_contour_selection_training'])
        jobpath = args.jobpath
        from FileUtil import gpfs2WinPath
        jobpath = gpfs2WinPath(jobpath)

    logger.initlogging(level='debug', logpath=os.path.join(jobpath, 'ContourSelect.log'))
    log = logger.getLogger('ContourSelect')
    log.info("Start run Contour Selection job: {}".format(jobpath))
    log.info(str(vars(args)))

    myjob = ContourSelJob(jobpath)
    myjob.run()
    log.info("Successfully complete Contour Selection job, please review job results at {}".format(jobpath))
    logger.closelogging('ContourSelect')

if __name__ == '__main__':
    main()