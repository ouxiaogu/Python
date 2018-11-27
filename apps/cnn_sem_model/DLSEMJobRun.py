# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-11-26 13:06:54

DL SEM job runner

Last Modified by:  ouxiaogu
"""

import argparse
from DLSEMJob import DLSEMJob

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../../libs/common")
from logger import logger

def main():
    parser = argparse.ArgumentParser(description='xml-driving CNN sem job')
    parser.add_argument('jobpath', help='cnn sem job path')
    try:
        args = parser.parse_args()
        jobpath = args.jobpath
    except:
        parser.print_help()
        # parser.exit()
        args = parser.parse_args(['./samplejob'])
        jobpath = args.jobpath
        print('No jobpath is inputed, use sample job path: %s' % jobpath)
    logger.initlogging(level='debug', logpath=os.path.join(jobpath, 'DLSEM.log'))
    log = logger.getLogger('DLSEM')
    log.info("Start run DLSEM job: {}".format(jobpath))
    log.info(str(vars(args)))

    myjob = DLSEMJob(jobpath)
    myjob.run()

    log.info("Successfully complete DLSEM job, please review job results at {}".format(jobpath))
    logger.closelogging('DLSEM')

if __name__ == '__main__':
    main()