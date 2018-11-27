# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-11-22 09:41:24

FEM+ D2DB correction main function

Last Modified by:  ouxiaogu
"""

import argparse
from D2DBCorrection import D2DBCorrection

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../libs/common")
from logger import logger

def main():
    parser = argparse.ArgumentParser(
        description='D2DB residual error correction in FEM+')
    parser.add_argument(
        'jobpath', type=str, help='FEM+ D2DB job path(job type: ModelCheck)')
    parser.add_argument(
        '-I', '--semspec', type=str, help='Input SEM Spec file')
    parser.add_argument(
        '-O', '--outspec', type=str, help='Output SEM Spec file')
    parser.add_argument(
        '-r', '--reuse_result', type=str,
        help='Whether to reuse the FEM+ pattern align correction job results',
        choices=['yes', 'no'], default='yes')
    parser.add_argument(
        '-u', '--reuse_lua', type=str,
        help='Whether to reuse the FEM+ pattern align correction job lua',
        choices=['yes', 'no'], default='no')

    try:
        args = parser.parse_args()
        jobpath = args.jobpath
        semspec = args.semspec.strip() if args.semspec is not None else ""
        outspec = args.outspec.strip() if args.outspec is not None else ""
        reuse_result = True if args.reuse_result.strip() == 'yes' else False
        reuse_lua = True if args.reuse_lua.strip() == 'yes' else False
    except:
        parser.print_help()
        # parser.exit()
        
        args = parser.parse_args(['/gpfs/DEV/FEM/peyang/release/E8.0/MOD9944/job2_2D_CD+EP_align_correction_GN_clone'])
        jobpath = args.jobpath
        from FileUtil import gpfs2WinPath
        jobpath = gpfs2WinPath(jobpath)
        print('No jobpath is inputed, use sample job path: %s' % jobpath)
        semspec, outspec, reuse_result, reuse_result = 4*[None]

    logger.initlogging(
        level='debug',
        logpath=os.path.join(jobpath, 'D2DBCorrection.log'))
    log = logger.getLogger('D2DBCorrection')
    log.info("Start to run FEM+ D2DB Correction job: {}".format(jobpath))
    log.info(str(vars(args)))

    correction = D2DBCorrection(jobpath, semspec, outspec, reuse_result, reuse_lua)
    correction.run()

    log.info(
        "Successfully complete FEM+ D2DB Correction job, please review job results at {}"
        .format(jobpath))
    logger.closelogging('D2DBCorrection')


if __name__ == '__main__':
    main()