# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-20 12:06:57

Tag contour, by drawing outlier bboxes 

Last Modified by:  ouxiaogu
"""

import numpy as np
import cv2

import sys
import os.path
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../../libs/tacx/")
from SEMContour import SEMContour

g_epslmt = 1e-9


def decideAngleType(angle):
    '''
    normalize angle into (-180, 180], then get angle types:

    np.linspace(22.5, 180-22.5, 4)
    Out[18]: array([  22.5,   67.5,  112.5,  157.5])

    np.linspace(22.5-180, -22.5, 4)
    Out[19]: array([-157.5, -112.5,  -67.5,  -22.5])

    in the range of interval [-22.5, 22.5) from center line
    
    angle type, direction, center lines

    * 0, H: 0, 180, -180
    * 1, +45: 45, -135
    * 2, V: 90, -90
    * 3, -45: 135, -45
    '''

    # angle into (-180, 180]
    angle = angle%360
    if angle > 180:
        angle = angle - 360

    angleType = 0
    if -22.5<=angle<22.5 or -22.5<=(angle-180)<22.5 or -22.5<=(angle+180)<22.5:
        angleType=0
    elif -22.5<=(angle-45)<22.5 or -22.5<=(angle+135)<22.5:
        angleType=1
    elif -22.5<=(angle-90)<22.5 or -22.5<=(angle+90)<22.5:
        angleType=2
    elif -22.5<=(angle-135)<22.5 or -22.5<=(angle+45)<22.5:
        angleType=3
    return angleType

class PolygonDrawer(object):
    def __init__(self, im, contour=None, window_name='Drawer'):
        self.raw = im
        if np.ndim(im) == 2:
            self.im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR);
        elif np.ndim(im) == 3:
            self.im = im
        else:
            sys.exit("image need to be an instance of numpy ndarray \n")
        if contour is None and not isinstance(contour, SEMContour):
            sys.exit("contour need to be an instance of SEMContour\n")
        self.window_name = window_name
        self._init_parms()

    def _init_parms(self):
        self.final = self.im.copy()
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.FINAL_LINE_COLOR = (0, 255, 255)
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
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with {} points.".format(len(self.points) ))
            print(str(self.points))
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
                # Draw all the current polygon segments
                cv2.polylines(self.im, np.array([self.points]), False, self.FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                cv2.line(self.im, self.points[-1], self.current, self.WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, self.im)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        # of a final polygon
        if (len(self.points) > 0):
            #cv2.fillPoly(self.im, np.array([self.points]), self.FINAL_LINE_COLOR)
            cv2.polylines(self.final, np.array([self.points]), True, self.FINAL_LINE_COLOR, 1)
        # And show it
        cv2.imshow(self.window_name, self.final)
        # Waiting for the user to press any key
        cv2.waitKey()
        cv2.destroyWindow(self.window_name)
        return self.final

    def getROIData(self, masked_mat=False):
        '''
        generate polygon ROI by self.points, access the raw image data under
        the polygon ROI
        '''
        roi=getPolyROI(self.raw, self.points, masked_mat)
        return roi

def getPolyROI(im, vertexes, masked_mat=False):
    '''
    Access the image data under the input vertexes defined polygon ROI.
    Parameters
    ----------
    im : 2D array like
        input image object
    vertexes : list of vertex
        vertex without self intersection, which define a polygon
    Returns
    -------
    dst : list
        roi data
    '''
    if(np.ndim(im) != 2):
        raise ValueError("getPolyROI only support 2D image, input im's shape is {}!\n".format(str(im.shape)))
    mask = np.zeros(im.shape, np.uint8)
    rc = np.array(vertexes)
    cv2.fillPoly(mask, [rc], 255)
    matrix = np.ma.array(im, mask=~(mask>128))
    if masked_mat:
        roi = matrix
    else:
        roi = matrix.compressed()
        # roi = matrix[~matrix.mask]
    return roi

class PointPairDrawer(PolygonDrawer):
    """PointPairDrawer: drawer for pairs of points"""
    def __init__(self, im, window_name, **kwargs):
        super(PointPairDrawer, self).__init__(im, window_name)
        self.pairs = []
        self.drawfunc = kwargs.get('drawfunc', None)

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
                lastpair = self._enforce_line_type([self.points[-2], self.points[-1]])
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

    def _enforce_line_type(self, src):
        '''
        Line type 0, 1, 2, 3
        - line_type 0: any direction, (x0, y0), (x1, y1)
        - line_type 1: Horizontal line, (x0, y0), (x1, y0)
        - line_type 2: Vertical line, (x0, y0), (x0, y1)
        - line_type 3: 45*N line, (x0, y0), (x1, y1')
        '''
        dst = src

        head, tail = src
        x0, y0 = head
        x1, y1 = tail
        angle = np.arctan2(y1-y0, x1-x0)
        angle_type = decideAngleType(angle)

        if angle_type == 0: # H
            dst[-1] = (x1, y0)
        elif self.line_type == 2: # V
            dst[-1] = (x0, y1)
        else: # 45*N
            tan_theta = np.copysign(1, y1-y0)/np.copysign(1, x1-x0)
            y1 = y0 + tan_theta*(x1-x0)
            dst[-1] = (x1, y1)
        return dst

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

class LineDrawer(PointPairDrawer):
    """LineDrawer: line by a pair of points, head&tail,
    support H,V,45,any line"""
    def __init__(self, im, window_name):
        kwargs = {'line_type': line_type}
        super(LineDrawer, self).__init__(im, window_name, **kwargs)

class RectangleDrawer(PointPairDrawer):
    """Rectangle is composed by a pair of point: tl and br"""
    def __init__(self, im, window_name):
        kwargs = {'drawfunc': cv2.rectangle}
        super(RectangleDrawer, self).__init__(im, window_name, **kwargs)
