# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-10-28 18:34:10

Contour Track-bar filter, interactively tune the filter thresh by track-bar, and review the result

Last Modified by:  ouxiaogu
"""

import numpy as np
import cv2
import cvui

import sys
import os.path
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../libs/tacx/")
from SEMContour import SEMContour, ContourBBox
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../libs/common/")
from logger import logger
log = logger.getLogger(__name__)

class ContourTrackbarFilter(object):
    """
    ContourFilterViaTrackbar: show SEM image & contour, user can interactively tune the filter thresh by track-bar, and review the result

    """
    def __init__(self, im, contour, window_name, colname='ridge_intensity', **kwargs):
        self.raw = im
        if np.ndim(im) == 2:
            self.im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif np.ndim(im) == 3:
            self.im = im
        else:
            sys.exit("image need to be an instance of numpy ndarray \n")
        self.contour = contour
        self.contourdf = contour.toDf()
        self.window_name = window_name
        self.colname = colname

        self._init_parms()

    def _init_parms(self):
        self.thres_range = [0., 0.3]
        self.thres = [0.]
        self.numTotal = 0
        self.numOutlier = 0
        self.colmax = [self.contourdf.max()[self.colname]]

        try:
            vmax = np.iinfo(self.im.dtype).max
        except ValueError:
            raise ValueError("Only support integer data type, input im dtype is {}".format(self.im.dtype))
        self.FINAL_OUTLIER_COLOR = (0, 0, vmax) # Red

    def printUsage(self):

        print("\n###############################################################\n"
            "Usage of ContourTrackbarFilter:\n\n"
            "User can tune the contour filter thresh via track-bar\n"
            "Background:              SEM image and raw contour\n"
            "color Red:               bad contour part(${attribute} < thresh)\n"

            "Trackbar Dragging:       Change the filter's relative threshold\n"
            "ESC:                     Exit drawing\n"
            "###############################################################\n")

    def loadFrameData(self):
        frame = self.im.copy()
        
        self.absThres = self.colmax[0] * self.thres[0]
        # absThres = self.thres[0]
        self.numTotal = len(self.contourdf)
        flt = self.contourdf[self.colname] < self.absThres
        badpoints = self.contourdf.loc[flt, ['polygonId', 'offsetx', 'offsety']]
        self.numOutlier = len(badpoints)

        thickness = 1
        if len(badpoints) > 0:
            contourPointsVec = []
            grouped = badpoints[['polygonId', 'offsetx', 'offsety']].groupby('polygonId')
            for name, group in grouped:
                contourPointsVec.append(group.loc[:, ['offsetx', 'offsety']].values.astype('int32'))
            frame = cv2.polylines(frame, contourPointsVec, False, self.FINAL_OUTLIER_COLOR, thickness)
            # frame = cv2.drawContours(frame, contourPointsVec , -1, self.FINAL_OUTLIER_COLOR, thickness)
        return frame

    def run(self):

        # Init cvui and tell it to create a OpenCV window, i.e. cv2.namedWindow(WINDOW_NAME).
        cvui.init(self.window_name)
        frame = np.zeros_like(self.im)
        imh, imw, _ = self.im.shape
        xini_wnd, yini_wnd = imw*0.04, imh*0.05

        while (True):
            frame = self.loadFrameData()

            # Render the settings window to house the checkbox
            # and the trackbars below.
            # cvui.window(frame, xini_wnd, yini_wnd, 100, 200, '{} filter threshold'.format(self.colname)) # 'Settings'
            cvui.beginColumn(frame, xini_wnd+1, yini_wnd+10, 200, 100, 6)
            
            # A trackbar to control the filter threshold values
            cvui.text(frame, xini_wnd+1, yini_wnd+10, '{} filter threshold'.format(self.colname))
            cvui.trackbar(frame, xini_wnd+1, yini_wnd+40, 250, self.thres, self.thres_range[0], self.thres_range[1], 4, '%.3Lf')
            cvui.space(10)
            cvui.text(frame, xini_wnd+1, yini_wnd+100, "{}/{} points is labeled as outlier".format(
                self.numOutlier, self.numTotal))

            cvui.endColumn()

            # This function must be called *AFTER* all UI components. It does
            # all the behind the scenes magic to handle mouse clicks, etc.
            cvui.update()

            # Show everything on the screen
            cv2.imshow(self.window_name, frame)

            # Check if ESC key was pressed
            if cv2.waitKey(20) == 27:
                break
        cv2.destroyAllWindows()

        return {self.colname: self.absThres}