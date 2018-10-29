# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-20 12:06:57

Tag contour, by drawing outlier bboxes 

Last Modified by:  ouxiaogu
"""

import numpy as np
import cv2
import collections
import re

import sys
import os.path
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../../libs/tacx/")
from SEMContour import SEMContour, ContourBBox
from MxpStage import MxpStage
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../../libs/imutil/")
from ImGUI import imread_gray

sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../../libs/common/")
from XmlUtil import addChildNode, getConfigData, setConfigData
import logger
log = logger.setup("ContourLabeling", 'debug')

class RectangleDrawer(object):
    """
    RectangleDrawer: drawer for rectangle, 
    Rectangle represented by pairs of points

    Example
    -------
    `pairs`:
    [[(481, 732), (523, 771)], [(481, 732), (522, 774)]]

    Where,
    - rect 1: [(481, 732), (523, 771)]
    - rect 2: [(481, 732), (522, 774)]

    """
    def __init__(self, im, window_name, **kwargs):
        self.raw = im
        if np.ndim(im) == 2:
            self.im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif np.ndim(im) == 3:
            self.im = im
        else:
            sys.exit("image need to be an instance of numpy ndarray \n")
        self.window_name = window_name
        self.drawfunc = kwargs.get('drawfunc', cv2.rectangle)

        self._init_parms()

    def _init_parms(self):
        self.final = self.im.copy()
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.pairs = []  # List of point pairs

        vmax = np.iinfo(self.im.dtype).max
        self.FINAL_OUTLIER_COLOR = (0, 0, vmax) # Red
        self.FINAL_GOOD_COLOR = (0, vmax, 0) # Green
        self.WORKING_LINE_COLOR = (vmax//2, vmax//2, vmax//2)

    def printUsage(self):
        print("\nUsage of RectangleDrawer:\n\n"
            "Rectangle composed by a pair of points:\n"
            "  - To input Good Contour Points area, needs: xini< xend and yini < yend;\n"
            "  - Otherwise, it's Outlier Area\n\n"
            "mouse odd  left click:   Rectangle top left point\n"
            "mouse move:              Rectangle bottom right point updating\n"
            "mouse even left click:   Rectangle bottom right point\n"
            "mouse right click:       Stop drawing rectangle\n"
            "ESC:                     Exit drawing\n")

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            sys.stdout.write("Adding point #{} with position({},{})!\n".format(len(self.points), x, y))
            self.points.append((x, y))
            if len(self.points)>=2 and len(self.points)%2 == 0:
                lastpair = [self.points[-2], self.points[-1]]
                self.pairs.append(lastpair)
                _, lastpnt = lastpair
                self.points[-1] = lastpnt

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing drawing with {} points.".format(len(self.points) ))
            print(str(self.points))
            print("Completing drawing with {} point pairs.".format(len(self.pairs) ))
            print(str(self.pairs))
            self.done = True

    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, self.im)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            if (len(self.points) > 0):
                # Draw all the lines
                for pair in self.pairs:
                    head, tail = pair
                    if head[0] < tail[0] and head[1] < tail[1]:
                        self.drawfunc(self.im, head, tail, self.FINAL_GOOD_COLOR)
                    else:
                        self.drawfunc(self.im, head, tail, self.FINAL_OUTLIER_COLOR)
                # And  also show what the current segment would look like
                if len(self.points) > 0:
                    cv2.line(self.im, self.points[-1], self.current, self.WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, self.im)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finished entering the points, so let's make the final drawing
        # of a final polygon
        for pair in self.pairs:
            head, tail = pair
            if head[0] < tail[0] and head[1] < tail[1]:
                self.drawfunc(self.final, head, tail, self.FINAL_GOOD_COLOR)
            else:
                self.drawfunc(self.final, head, tail, self.FINAL_OUTLIER_COLOR)
        # And show it
        cv2.imshow(self.window_name, self.final)
        # Waiting for the user to press any key
        cv2.waitKey()
        cv2.destroyWindow(self.window_name)
        return self.final

    def getROICoord(self):
        return self.pairs

class ContourSelLabelStage(MxpStage):
    """
    ContourSelLabelStage

    Contour Select App: labeling data stage

    Traverse Patterns with cost>0, based on pattern image and contour, to draw
    Outlier BBox by human best effort, the contour points inside any outlier 
    bbox will be considered as 'Outlier', add a 'UserLabel' to record:

        - 1: means `good`, not in any Outlier bbox
        - 0: means `bad`, at least in 1 Outlier bbox

    the labeled data will be used for model calibration/verification, the 
    UserLabel is ground truth.

    TODO, if without this Contour Labeling Stage, use the MXP_Flag as UserLabel,
    MXP_Flag with filter BitMask to label 'good' & 'bad'.
    """
    tgtColName = 'UserLabel'

    def selectLabelingPatternsRandomly(self):
        start_pattern_name = getConfigData(self.d_cf, 'start_pattern_name', 0)
        nsamples = getConfigData(self.d_cf, 'samples', -1)
        allpatternids = []
        validpatternids = []
        
        # get pattern names
        for idx, occf in enumerate(self.d_ocf.findall('.pattern')):
            if getConfigData(occf, 'costwt') <= 0:
                continue
            pid = getConfigData(occf, 'name')
            try:
                if int(pid) > start_pattern_name:
                    validpatternids.append(pid)
            except:
                pass
            allpatternids.append(pid)

        # randomly choose sample patterns
        selectedpatternids = allpatternids
        if len(validpatternids) != 0:
            nsamples = min(len(validpatternids), nsamples)
        else:
            nsamples = min(len(allpatternids), nsamples)
        if nsamples > 0:
            samplepatternids = allpatternids
            if len(validpatternids) >= nsamples:
                samplepatternids = validpatternids

            np.random.seed(128)
            np.random.shuffle(samplepatternids)
            selectedpatternids = samplepatternids[:nsamples]
        return selectedpatternids

    def selectLabelingPatterns(self):
        pattern_selection_mode = getConfigData(self.d_cf, '.select_sample/mode', 'random')
        if pattern_selection_mode == 'explicit':
            selectedpatternids = getConfigData(self.d_cf, '.select_sample/filter', '')
            selectedpatternids = [c.strip() for c in selectedpatternids.split(",")]
        elif pattern_selection_mode == 'regex':
            flt = getConfigData(self.d_cf, '.select_sample/filter', '*')
            selectedpatternids = flt
        else:
            selectedpatternids = self.selectLabelingPatternsRandomly()
        return selectedpatternids

    def run(self):
        selectedpatternids = self.selectLabelingPatterns()
        log.info("pattern samples to be labeled: {}".format(selectedpatternids))

        for idx, occf in enumerate(self.d_ocf.findall('.pattern')):
            if getConfigData(occf, 'costwt') <= 0:
                continue
            patternid = getConfigData(occf, '.name')
            if isinstance(selectedpatternids, collections.Iterable) and patternid not in selectedpatternids:
                continue
            elif isinstance(selectedpatternids, str) and not re.match(selectedpatternids, patternid):
                continue

            # load image and contour
            imgfile = os.path.join(self.jobresultabspath, getConfigData(occf, '.image/path'))
            contourfile = os.path.join(self.jobresultabspath, getConfigData(occf, '.contour/path'))
            im, rawcontour, origin = self.loadPatternData(imgfile, contourfile)

            # load image and contour
            drawer = RectangleDrawer(im, "Pattern {} Contour Data Labeling...".format(patternid))
            drawer.printUsage()
            drawer.run()
            rectcoord = drawer.getROICoord()

            # save bbox into occf
            bboxcf = addChildNode(occf, 'bbox')
            goodContourAreas = []
            outlierAreas = []
            idxGoodRect, idxOutlierRect = 0, 0
            for rect in rectcoord:
                tl, br = rect
                xini, yini = map(np.add, tl, origin) # compensate the origin
                xend, yend = map(np.add, br, origin)

                bboxstr = "{}, {}, {}, {}".format(xini, yini, xend, yend)
                if (xini < xend) and (yini < yend):
                    setConfigData(bboxcf, 'Good', val=bboxstr, count=idxGoodRect)
                    goodContourAreas.append((xini, yini, xend, yend))
                    idxGoodRect += 1
                else:
                    setConfigData(bboxcf, 'Outlier', val=bboxstr, count=idxOutlierRect)
                    xmin, xmax = min(xini, xend), max(xini, xend)
                    ymin, ymax = min(yini, yend), max(yini, yend)
                    outlierAreas.append((xmin, ymin, xmax, ymax))
                    idxOutlierRect += 1
            log.debug("#Good contour areas: {}\n{}".format(len(goodContourAreas), goodContourAreas))
            log.debug("#Outlier contour areas: {}\n{}".format(len(outlierAreas), outlierAreas))

            # apply bbox into contour files
            labeledconour = self.labelPatternContour(rawcontour, goodContourAreas, outlierAreas)
            newcontourfile_relpath = os.path.join(self.stageresultrelpath, '{}_image_contour.txt'.format(patternid))
            newcontourfile = os.path.join(self.jobresultabspath, newcontourfile_relpath)
            labeledconour.saveContour(newcontourfile)
            setConfigData(occf, '.contour/path', newcontourfile_relpath)

    def loadPatternData(self, imgfile='', contourfile=''):
        bSucceedReadCt = False

        # read contour
        contour = SEMContour()
        bSucceedReadCt = contour.parseFile(contourfile)
        if not bSucceedReadCt:
            raise OSError("Error, contourfile('{}') cannot be parsed".format(contourfile))
        xini, yini, xend, yend = contour.getBBox()
        contourPointsVec = [] # depth of 3
        for polygon in contour.getPolygonData():
            contourpoints = []
            for point in polygon['points']:
                contourpoints.append([point[0]-xini, point[1]-yini]) # origin as (xini, yini)
            contourPointsVec.append(np.around(contourpoints).astype(int))
        log.debug("contour points shape {} 1st point {}".format(contourPointsVec[0].shape, contourPointsVec[0][0]))
            
        # read image
        try:
            im, _ = imread_gray(imgfile)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        except:
            imw, imh = contour.getshape()
            imw, imh = int(imw), int(imh)
            im = 255//2 * np.ones((imh, imw, 3), dtype=np.uint8) # gray background

        # overlay image & contour
        thickness = 1
        vmax = np.iinfo(im.dtype).max
        CONTOUR_COLOR = (0, vmax, vmax) # yellow
        # im = cv2.drawContours(im, contourPointsVec, -1, CONTOUR_COLOR, thickness)
        im = cv2.polylines(im[yini:yend, xini:xend], contourPointsVec, False, CONTOUR_COLOR, thickness)

        return im, contour, (xini, yini)

    @staticmethod
    def updateContourROI(contour, im=None, mode='crop', overlay=True):
        '''
        Update contour ROI, mode could be 'crop' or 'extend'

        Parameters
        ----------
        mode:   string
            * 'crop': crop the image into contour bbox, output `contour point cords -= (xini, yini)`
            * 'crop': extend the image into full size, output `contour point cords += (xini, yini)`
        overly: boolean
            * 'True': output image is the overlay of image + contour
            * 'False': output image is just image itself
        
        Returns
        -------
        outim: image object
            could be overlay of image + contour or image itself
        outcontour: SEMContour object
            contour file will different cords
        '''
        outim, outcontour = None, None
        # process image
        if im is None:
            log.warning("input image is None, use gray background")
            imw, imh = contour.getshape()
            imw, imh = int(imw), int(imh)
            im = 255//2 * np.ones((imh, imw, 3), dtype=np.uint8) # gray background
        else:
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            if not np.integer(im):
                raise TypeError("only support int type images, input image type: {}".format(im.dtype))

        vmax = np.iinfo(im.dtype).max
        CONTOUR_COLOR = (0, vmax, 0) # green

        # process contour, and overlay image
        df = contour.toDf()
        xini, yini, xend, yend = contour.getBBox()
        if mode == 'crop':
            outim = im[yini:yend, xini:xend]

            df.loc['offsetx', :] -= xini
            df.loc['offsety', :] -= yini
            outcontour = contour.fromDf(df)
            if overlay:
                contourPointsVec = []
                grouped = df[['polygonId', 'offsetx', 'offsety']].groupby('polygonId')
                for name, group in grouped:
                    contourPointsVec.append(group.loc[['offsetx', 'offsety']].values)
                thickness = 1
                outim = cv2.polylines(outim, contourPointsVec, False, CONTOUR_COLOR, thickness)  
        elif mode == 'extend':
            df.loc['offsetx', :] += xini
            df.loc['offsety', :] += yini
            outcontour = contour.fromDf(df)
        else:
            raise ValueError("Only support mode of 'crop' or 'extend', input is {}".format(mode))

        return outim, outcontour

    def labelPatternContour(self, contour, goodContourAreas, outlierAreas):
        columnTitle = contour.getColumnTitle()
        columnTitle.append(self.tgtColName) # UserLabel
        contour.setColumnTitle(columnTitle)
        
        polygons = contour.getPolygonData()
        for polygon in polygons:
            for j, point in enumerate(polygon['points']):
                label = np.nan

                coord = [point[0], point[1]]
                if any([ContourBBox(*rect).contains(coord) for rect in outlierAreas]): # outlierAreas have higher priority than goodContourAreas
                    label = 0 # 'bad'
                elif any([ContourBBox(*rect).contains(coord) for rect in goodContourAreas]):
                    label = 1 # 'good'
                polygon['points'][j].append(label)
        contour.setPolygonData(polygons)
        return contour








