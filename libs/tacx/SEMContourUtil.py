# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-12-01 12:34:26

Some utility functions for SEMContour class

Last Modified by:  ouxiaogu
"""
from SEMContour import SEMContour
import numpy as np

import sys, os
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../common/")
from logger import logger
# logger.initlogging('debug')
log = logger.getLogger(__name__)

def connectDiscreteSubSegmentPerContour(contourdf, gapTolenrance=2, minSubSegLength=4, inplace=False):
    '''
    After contour selection, there are two types of discrete contour sub-segment:
        - discrete removed-points among kept-points
        - discrete kept points among removed-points
    Here, we will connect the discrete sub-segment by change points in the gap into 
    the same Label if gap <= `gapTolenrance`;
    At the same time, record the length of connected sub-segment, in which the 
    points should have the same labels, if length < `minSubSegLength`, all 
    change into opposite label, unless the sub-segment is located at the head/tail
    of current segment.

    Parameters:
    -----------
    contourdf: [in, out] contour as DataFrame
        [in] Contour df, must contains `polygonId`, `ClfLabel`
        [out] Contour df: if inplace=False, add `NewLabel`; otherwise, directly 
            update the revised new label into `ClfLabel`
    gapTolenrance: int 
        if gap <= `gapTolenrance`, it could be changed into opposite Label
    minSubSegLength: int
        if length < `minSubSegLength`, all its point change into opposite label,
        unless the sub-segment is located at the head/tail of current segment.
    '''

    grouped = contourdf.groupby('polygonId')
    colname = 'ClfLabel' if inplace else 'NewLabel'
    for polygonId, group in grouped:
        # log.debug("segment {}, length {}".format(polygonId, len(group)))
        segLabels = group.loc[:, 'ClfLabel'].values
        newLabels = connectDiscreteSubSegmentPerSeg(segLabels, gapTolenrance, minSubSegLength, debugPrefix=polygonId)
        contourdf.loc[contourdf['polygonId']==polygonId, colname] = newLabels
        kept2removed = [segLabels[i]==1 and newLabels[i]==0 for i in range(len(segLabels))]
        removed2kept = [segLabels[i]==0 and newLabels[i]==1 for i in range(len(segLabels))]
        if any(kept2removed) or any(removed2kept):
            log.debug("segment {}, length {}, kept2removed {}, removed2kept {}".format(polygonId, len(segLabels), sum(kept2removed), sum(removed2kept)))

def connectDiscreteSubSegmentPerSeg(segLabels, gapTolenrance=2, minSubSegLength=4, debugPrefix=''):
    # log.debug('subSeg {} length {}, removed {}'.format(debugPrefix, len(segLabels), len(segLabels)-sum(segLabels)))
    if sum(segLabels) >= (len(segLabels) - gapTolenrance):
        return  len(segLabels)*[int(np.round(1.0*sum(segLabels)/len(segLabels)))]

    # step 1, search from head
    newLabels = []
    idx = 0
    gapSeg = []
    while idx < len(segLabels):
        subSeg = gapSeg + [idx]
        subHeadIdx = subSeg[0]
        del gapSeg[:] # gapSeg.clear()
        # log.debug("subHeadIdx {} , label {}".format(subHeadIdx, segLabels[subHeadIdx]))

        if segLabels[subSeg[0]] != segLabels[subSeg[-1]]: # gapSeg not the same with 1st join point
            gapSeg.append(idx)
            subSeg.pop()
        while True: # new subSeg tracing
            idx += 1
            if idx == len(segLabels): # end of current vline
                subSeg += gapSeg # gap always later than existed subSeg
                del gapSeg[:] # gapSeg.clear()
                break
            if segLabels[idx] == segLabels[subHeadIdx]: # gap terminate
                subSeg += gapSeg
                subSeg.append(idx)
                del gapSeg[:] # gapSeg.clear()
            else: # gap continue
                gapSeg.append(idx)
                if len(gapSeg) > gapTolenrance: # too large gap
                    break

        # revise ClfLabel for current sub segment
        if len(subSeg) < minSubSegLength: # subSeg is too short
            if (subSeg[0] == 0) or (subSeg[-1] == len(segLabels)-1): # head or tail, use its label
                newLabels += len(subSeg) * [segLabels[subHeadIdx]]
            else: # opposite label
                newLabels += len(subSeg) * [int(not segLabels[subHeadIdx])]
        else: # subSeg is with normal length
            newLabels += len(subSeg) * [segLabels[subHeadIdx]]
        # log.debug('current subSeg {} ~ {}, add len {}, newLabels length {}'.format(subSeg[0], subSeg[-1], len(subSeg), len(newLabels)))
        # traverse next sub segment
        del subSeg[:] # subSeg.clear()
        idx += 1
    if len(gapSeg) != 0:
        newLabels += [segLabels[idx] for idx in gapSeg]
    # change newLabelsFromTail to the same order with newLabelsFromHead
    newLabelsFromHead = newLabels[:] 

    # step 2, search from tail
    newLabels = []
    idx = len(segLabels) - 1
    gapSeg = []
    while idx >= 0:
        subSeg = gapSeg + [idx]
        del gapSeg[:] # gapSeg.clear()
        subHeadIdx = subSeg[0]
        # log.debug("subHeadIdx {} , label {}".format(subHeadIdx, segLabels[subHeadIdx]))

        if segLabels[subSeg[0]] != segLabels[subSeg[-1]]: # gapSeg not the same with 1st join point
            gapSeg.append(idx) # new gap
            subSeg.pop()
        # new subSeg tracing
        while True:
            idx -= 1
            if idx < 0: # end of current vline
                subSeg += gapSeg
                del gapSeg[:] # gapSeg.clear()
                break
            if segLabels[idx] == segLabels[subHeadIdx]: # gap terminate
                subSeg += gapSeg
                subSeg.append(idx)
                del gapSeg[:] # gapSeg.clear()
            else: # gap continue
                gapSeg.append(idx)
                if len(gapSeg) > gapTolenrance: # too large gap
                    break

        # revise ClfLabel for current sub segment
        if len(subSeg) < minSubSegLength: # subSeg is too short
            if (subSeg[0] == 0) or (subSeg[-1] == len(segLabels)-1): # head or tail, use its label
                newLabels += len(subSeg) * [segLabels[subHeadIdx]]
            else: # opposite label
                newLabels += len(subSeg) * [int(not segLabels[subHeadIdx])]
        else: # subSeg is with normal length
            newLabels += len(subSeg) * [segLabels[subHeadIdx]]
        # log.debug('current subSeg {} ~ {}, add len {}, newLabels length {}'.format(subSeg[0], subSeg[-1], len(subSeg), len(newLabels)))
        # traverse next sub segment
        del subSeg[:] # subSeg.clear()
        idx -= 1
    if len(gapSeg) != 0:
        newLabels += [segLabels[idx] for idx in gapSeg]

    newLabelsFromTail = newLabels[::-1]

    # step 3, pick the newLabels, which has the largest number of removed points

    # kept2removed = [segLabels[i]==1 and newLabelsFromHead[i]==0 for i in range(len(segLabels))]
    # removed2kept = [segLabels[i]==0 and newLabelsFromHead[i]==1 for i in range(len(segLabels))]
    # if any(kept2removed) or any(removed2kept):
    #     log.debug("segment {} searched, length {}, search from head, kept2removed {}, removed2kept {}".format(debugPrefix, len(segLabels), sum(kept2removed), sum(removed2kept)))

    # kept2removed = [segLabels[i]==1 and newLabelsFromTail[i]==0 for i in range(len(segLabels))]
    # removed2kept = [segLabels[i]==0 and newLabelsFromTail[i]==1 for i in range(len(segLabels))]
    # if any(kept2removed) or any(removed2kept):
    #     log.debug("segment {} searched, length {}, search from tail, kept2removed {}, removed2kept {}".format(debugPrefix, len(segLabels), sum(kept2removed), sum(removed2kept)))

    try:
        assert(len(newLabelsFromHead) == len(newLabelsFromTail) == len(segLabels))
    except AssertionError:
        raise AssertionError("segment {}, len of newLabelsFromHead newLabelsFromTail segLabels: {} {} {}".format(debugPrefix, len(newLabelsFromHead), len(newLabelsFromTail), len(segLabels)))


    if (len(newLabelsFromHead) - sum(newLabelsFromHead)) > (len(newLabelsFromTail) - sum(newLabelsFromTail)):
        return newLabelsFromHead
    else:
        return newLabelsFromTail

if __name__ == '__main__':
    # contourfile = r'./3658_contour_selected.txt'
    contourfile = r'./439_image_contour.txt'
    contour = SEMContour()
    contour.parseFile(contourfile)
    contourdf = contour.toDf()
    # connectDiscreteSubSegment(contourdf)
    connectDiscreteSubSegmentPerContour(contourdf)
    dst = contour.fromDf(contourdf)
    from FileUtil import splitFileName
    dirname, filelabel, extension = splitFileName(contourfile)
    dst.saveContour(os.path.join(dirname, filelabel+'_new.'+extension))