# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-20 12:06:57

Tag contour, by drawing outlier bboxes 

Last Modified by: ouxiaogu
"""

import numpy as np
import cv2

import sys
import os.path
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../../libs/tacx/")
from SEMContour import SEMContour
from MxpStage import MxpStage

sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../../libs/common/")
from
import logger
log = logger.setup("ContourLabeling")

g_epslmt = 1e-9

class RectangleDrawer(object):
    """RectangleDrawer: drawer for pairs of points"""
    def __init__(self, im, window_name, **kwargs):
        self.raw = im
        if np.ndim(im) == 2:
            self.im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR);
        elif np.ndim(im) == 3:
            self.im = im
        else:
            sys.exit("image need to be an instance of numpy ndarray \n")
        self.window_name = window_name
        self._init_parms()

        self.pairs = []
        self.drawfunc = kwargs.get('drawfunc', cv2.rectangle)

    def _init_parms(self):
        self.final = self.im.copy()
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.FINAL_LINE_COLOR = (0, 0, 255) # Red
        self.WORKING_LINE_COLOR = (127, 127, 127)

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
                    self.drawfunc(self.im, head, tail, self.FINAL_LINE_COLOR)
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
            self.drawfunc(self.final, head, tail, self.FINAL_LINE_COLOR)
        # And show it
        cv2.imshow(self.window_name, self.final)
        # Waiting for the user to press any key
        cv2.waitKey()
        cv2.destroyWindow(self.window_name)
        return self.final

    def getROIData(self, masked_mat=False):
        '''
        generate rectangle/line ROI by self.pairs, access the raw image data
        under the corresponding ROI
        '''
        roidata = getROIByPointPairs(self.raw, self.pairs, self.drawfunc, masked_mat)
        return roidata

    def getROICoord(self):
        return self.pairs

def getROIByPointPairs(im, pairs, drawfunc, masked_mat=False):
    '''get ROI by point pairs, a pair of points can generate a rectangle or
    a line'''
    if(np.ndim(im) != 2):
        raise ValueError("getROIByPointPairs only support 2D image, input im's shape is {}!\n".format(str(im.shape)))
    mask = np.zeros(im.shape, np.uint8)
    thickness = 1
    if drawfunc == cv2.rectangle:
        thickness = cv2.FILLED
    for pair in pairs:
        head, tail = pair
        drawfunc(mask, head, tail, 255, thickness)
    matrix = np.ma.array(im, mask=~(mask>128))
    if masked_mat:
        roidata = matrix
    else:
        roidata = matrix.compressed()
        # roidata = matrix[~matrix.mask]
    return roidata

class ContourSelLabelStage(MxpStage):
    """
    ContourSelLabelStage

    Contour Select App: labeling data stage

    Traverse Patterns with cost>0, based on pattern image and contour, labeling
    the Outlier BBox, the contour points inside current bbox will be considered
    'Outlier', add a 'UserLabel' columns as 'bad', others considered as 'good', 
    the labeled data will be used for model calibration/verification, the 
    UserLabel is ground truth.

    TODO, if without this Contour Labeling Stage, use the MXP_Flag as UserLabel,
    MXP_Flag with filter BitMask to label 'good' & 'bad'.
    """

    def run(self):

        for idx, series in self.d_df.iterrows():
            if series.loc['cost'] <= 0:
                continue

            
