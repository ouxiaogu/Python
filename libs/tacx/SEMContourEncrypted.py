# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-12-06 10:04:19

SEM contour file is Encrypted since 19.02, this class is the conversion 
from SEMContourEncrypt into SEMContour.

*** Note ****
1. This pacakge is dependent on MXP binary's python api.
2. MXP binary's python api .pyc .pyo file is based on python2 
3. MXP binary's python api .so can only be recognized in Linux, don't support cross-platform yet

Last Modified by:  ouxiaogu
"""

from SEMContour import SEMContour

import sys, os
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../common/")
from logger import logger
from FileUtil import gpfs2WinPath
from PlatformUtil import inLinux
if inLinux():
    sys.path.insert(0, gpfs2WinPath("/gpfs/DEV/ATD/chezhang/SEM_Image/Tachyon_ImageAnalysis/ToolBox/MXP"))
    from MXPlibs.semutil import core as semutil_core
#from MXPlibs.utils import core as utils_core
# logger.initlogging('debug')
log = logger.getLogger(__name__)

def parseContourWrapper(contourfile):
    if contourfile is None:
        return None
    contour = SEMContour()
    if not (contour.parseFile(contourfile)):
        log.error("Can't parse contour file {} as unencrypted".format(contourfile))
        contour = None
        if inLinux(): # don't support cross-platform in Windows yet
            contourEcrypt = SEMContourEncrypted()
            if (contourEcrypt.parseFile(contourfile)):
                contour = contourEcrypt.contour
            else:
                log.error("Can't parse contour file {} neither as unencrypted nor encrypted".format(contourfile))
                contour = None
    if contour is not None:
        if contour.polygonNum == 0: # in case pattern contour file exists, but it's empty
            log.warning('Empty contour file at {}'.format(contourfile))
            contour = None
        df = contour.toDf()
        if any(df.columns.duplicated(keep='last')):
            log.debug("contour file header has duplicated column name: {}\nMost likely, performed multiple manual labeling for the same pattern, use the last one".format(df.columns.values))
            df = df.loc[:, ~df.columns.duplicated(keep='last')]
            contour = contour.fromDf(df)
    return contour

class SEMContourEncrypted(SEMContour):
    def __init__(self):
        super(SEMContourEncrypted, self).__init__()
        
    def parseFile(self, contourfile):
        self.contourEncyptedfile = contourfile
        self.contourEncypted = semutil_core.Contour(contourfile)
        try:
            self.contour = self.decrypt()
        except:
            log.warning("Can't parse Encrypted contour file {}".format(contourfile))
            return False
        return True

    def decrypt(self):
        dst = SEMContour()
        dst.setVersion(self.contourEncypted.getVersion())
        dst.setHeadTitle(self.contourEncypted.getHeader())
        dst.setColumnTitle(self.contourEncypted.getInputKeys())
        dst.setshape([self.contourEncypted.getWidth(), self.contourEncypted.getHeight()])
        dst.setPixel(self.contourEncypted.getPixel())
        dst.setOffset(self.contourEncypted.getCenterXY())
        bbox_cpp = self.contourEncypted.getContourBBox()
        bbox_py = list(bbox_cpp.getInitPoint()) + list(bbox_cpp.getEndPoint())
        dst.setBBox(bbox_py)

        # dst.polygonNum = self.contourEncypted.numOfSegments()
        d_contSegs= self.contourEncypted.getContour()
        polygonData =[None for i in range(len(d_contSegs))]
        for iseg in range(len(d_contSegs)):
            polygonData[iseg] = {}
            polygonData[iseg]['vertexNum'] = len(d_contSegs[iseg])
            polygonData[iseg]['polygonId'] = iseg
            polygonData[iseg]['polygon0Hole1'] = 0
            polygonData[iseg]['points'] =[]
            lines=[]
            for ipoint in range(len(d_contSegs[iseg])):
                tempmap = d_contSegs[iseg][ipoint].convertToMap()
                temp=[]
                for key in dst.getColumnTitle():
                    temp.append(tempmap[key])
                lines.append(temp)
            polygonData[iseg]['points'].extend(lines)
        dst.setPolygonData(polygonData)
        return dst

if __name__ == '__main__':
    contourfile = r'/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/IMEC/Case02_calaveras_v3/3Tmp/ContourSelection/020_AEI_contour_selection_training/h/data/dummydb/MXP/job1/ContourExtraction400result1/1_image_contour.txt'
    contourfile = gpfs2WinPath(contourfile)
    contour = parseContourWrapper(contourfile)