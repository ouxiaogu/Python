"""
-*- coding: utf-8 -*-
Created: peyang, 2018-01-24 21:07:29

Last Modified by: peyang

SEM: class to hold SEM Spec Table

Generate SEM spec table from:
    * MXP D2DB out xml: 'name','img_pixel','costwt',
                    'offset_x','offset_y'
    * iCal ALIGN_EDGE log: "Aligned Spec", with contain condition information

"""

from StringIO import StringIO
import pandas as pd
from MXPJob import MXPOutXml
import os.path
from TachyonJob import Job
from StrUtil import parseKW
from subprocess import check_output
import string
import re

DEFAULT_SPEC =\
'''SEM FILE    PIXEL   CENTER_X    CENTER_Y    COST_WT EDGE_FILE   DETECT_EDGE XAXIS_SWAP  YAXIS_SWAP  ROTATION    THETA   ALIGN_EDGE  V_REGION    GAUGE_TABLE GEOM_MAP    SCALE_XY_RATIO
1   1_image.pgm 1   0 0  1   1_image_contour.txt  1   0   0   0   0   0   -1,-1;-1,-1 -1  -1  -1'''

class SEMSpec(object):
    """docstring for SEMSpec"""
    def __init__(self, arg=None):
        super(SEMSpec, self).__init__()
        self.arg = arg
        self.DefaultSpec()

    def DefaultSpec(self):
        DATA = StringIO(DEFAULT_SPEC)
        df = pd.read_csv(DATA, delim_whitespace=True)  # sep="\s+"
        self.DefaultRow = df

    def from_mxp_occfs(self, src):
        """
        read out SEM spec from all patterns' out config

        args:
            src:    DataFrame, source table, come from oxml of patterns, contained name, offset_x, offset_y
        return:
            dst:    DataFrame, SEM spec table
        """
        RENAME_MAP={'name': 'SEM', 'img_pixel': 'PIXEL', 'costwt': 'COST_WT',
                    'offset_x': 'CENTER_X', 'offset_y': 'CENTER_Y'}
        MXP_EDIT_COLS = ['FILE', 'EDGE_FILE', 'DETECT_EDGE', 'ALIGN_EDGE']
        DETECT_EDGE = 0
        ALIGN_EDGE = 0

        spec = src[RENAME_MAP.keys()].rename(columns=RENAME_MAP)

        '''MXP_EDIT_COLS'''
        for col in MXP_EDIT_COLS:
            if col == 'FILE':
                postfix = '_image.pgm'
                spec.ix[:, col] = spec['SEM'].map(lambda x: str(int(x))+postfix)
            elif col == 'EDGE_FILE':
                postfix = '_image_contour.txt'
                spec.ix[:, col] = spec['SEM'].map(lambda x: str(int(x))+postfix)
            elif col == 'DETECT_EDGE':
                spec.ix[:, col] = DETECT_EDGE
            elif col == 'ALIGN_EDGE':
                spec.ix[:, col] = ALIGN_EDGE

        '''MXP unchanged cols: use default value'''
        DEFAULT_VAL_COLS = [col for col in self.DefaultRow.columns if col not in (MXP_EDIT_COLS + RENAME_MAP.values())]
        for col in DEFAULT_VAL_COLS:
            spec.ix[:, col] = self.DefaultRow.ix[0, col]

        spec = spec[self.DefaultRow.columns] # reorder spec columns
        self.spec = spec
        return spec

    def from_mxp_ocf(self, xmlfile):
        """
        generate SEM Spec from mxp out xml file

        example:
            xmlfile:
                <pattern>
                    <costwt>1</costwt>
                    <img_pixel>1</img_pixel>
                    <name>1</name>
                    <offset_x>2601989</offset_x>
                    <offset_y>333648</offset_y>
                </pattern>
                <pattern>...</pattern>
                ...
            xmlfile will convert to pattern occfs DataFrame
            then call from_mxp_occfs to generate the desired SEM Spec table
        """
        mxpocf = MXPOutXml(xmlfile)
        src = mxpocf.getoccfs()
        spec = self.from_mxp_occfs(src)
        return spec

    def from_ical_result(jobpath):
    # TODO
        pass



if __name__ == '__main__':
    from FileUtil import gpfs2WinPath

    INFILE = r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/Calaveras_v2/peyang/jobs/8GF02/06_study_c2c_id2db_v0_issuepatterns/h/cache/dummydb/result/MXP/job1/imaged2dbalignment600out.xml'

    INFILE = gpfs2WinPath(INFILE)
    SEM = SEMSpec()
    spec = SEM.from_mxp_ocf(INFILE)
    print spec