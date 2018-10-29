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
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../../libs/tacx/")
from SEMContour import SEMContour, ContourBBox
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../../libs/common/")
import logger
log = logger.setup("ContourTrackbarFilter", 'debug')

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
        self.colmax = [self.contourdf.max()[self.colname]]

        try:
            vmax = np.iinfo(self.im.dtype).max
        except ValueError:
            raise ValueError(" Invalid integer data type, im dtype {}".format(self.im.dtype))
        self.FINAL_OUTLIER_COLOR = (0, 0, vmax) # Red

        self.printUsage()

    def printUsage(self):
        print("\nUsage of ContourTrackbarFilter:\n\n"
            "User can tune the contour filter thresh via track-bar\n"
            "Background:              SEM image\n"
            "color Red:               bad contour part(${attribute} < thresh)\n"
            "color Green:             good contour part(${attribute} >= thresh)\n"

            "Trackbar Dragging:       Change the filter's relative threshold\n"
            "ESC:                     Exit drawing\n")

    def loadFrameData(self):
        frame = self.im.copy()
        
        absThres = self.colmax[0] * self.thres[0]
        flt = self.contourdf[self.colname] >= absThres
        badpoints = self.contourdf.loc[~flt, ['polygonId', 'offsetx', 'offsety']]

        thickness = 1
        if len(badpoints) > 0:
            contourPointsVec = []
            grouped = badpoints[['polygonId', 'offsetx', 'offsety']].groupby('polygonId')
            for name, group in grouped:
                contourPointsVec.append(group.loc[['offsetx', 'offsety']].values.astype(int32))
            frame = cv2.drawContours(frame, contourPointsVec , -1, self.FINAL_OUTLIER_COLOR, thickness)
        return frame

    def run(self):

        # Init cvui and tell it to create a OpenCV window, i.e. cv2.namedWindow(WINDOW_NAME).
        cvui.init(self.window_name)
        frame = np.zeros_like(self.im)

        while (True):
            frame = self.loadFrameData()

            # Render the settings window to house the checkbox
            # and the trackbars below.
            cvui.window(frame, 10, 50, 100, 100, 'Settings')

            # A trackbar to control the filter threshold values
            cvui.text('{} filter threshold'.format(self.colname))
            cvui.trackbar(frame, 15, 80, 165, self.thres, self.thres_range[0], self.thres_range[1], 4, '%.3Lf')

            # Exit the application if the quit button was pressed.
            # It can be pressed because of a mouse click or because 
            # the user pressed the 'q' key on the keyboard, which is
            # marked as a shortcut in the button label ('&Quit').
            # if cvui.button('&Quit'):
                # break

            # This function must be called *AFTER* all UI components. It does
            # all the behind the scenes magic to handle mouse clicks, etc.
            # cvui.update()

            # Show everything on the screen
            cv2.imshow(self.window_name, frame)


            # Check if ESC key was pressed
            if cv2.waitKey(20) == 27:
                break

        return self.thres[0]