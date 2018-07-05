"""
-*- coding: utf-8 -*-
Created: ouxiaogu, 2018-07-03 11:19:46

Last Modified by: ouxiaogu
"""

import numpy as np
import re

__all__ = ['readDumpImage', 'readBBox', 'gen_multi_image_overview']

def readDumpImage(infile, skip_header=0):
    im = np.genfromtxt(infile, skip_header=skip_header).astype('float32')
    imat = np.asmatrix(im)
    return imat

def readBBox(infile):
    bbox = None
    with open(infile) as f:
        for i, line in enumerate(f.readlines()):
            if i > 2:
                break
            m = re.search("^BBox: xini = (\d+), xend = (\d+), yini = (\d+), yend = (\d+)", line)
            if m is not None:
                bbox = m.groups()
                break
    return bbox

def gen_multi_image_overview(src_dir, reg_pattern=None, title=None, im_name=None):
    '''generate a overview for multiple image with image title

    command:

    montage -label Balloon   balloon.gif  \
            -label Medical   medical.gif  \
            -tile 4x  -frame 5  -geometry '300x300+2+2>' \
            import os.path
            -title 'My Images'     titled.jpg
    '''
    fsc = FileScanner(src_dir)
    files = fsc.scan_files(regex_pattern=reg_pattern)
    files.sort()
    labels = getFileLabel(files)
    strLabels = ''
    for ix, label in enumerate(labels):
        strLabels += "-label {} {} ".format(label, files[ix])
    name = "overview" if im_name is None else im_name
    outfile = os.path.join(src_dir, name+'.jpg')
    if title:
        outfile = "-title '{}'",format(title) + outfile
    command = ''' montage {} \
            -tile 4x  -frame 5  -geometry '300x300+2+2>' \
            {}'''.format(strLabels, outfile)
    log.info(command)
    call(command, shell=True)
    log.info("please find overview image at {}".format(outfile))