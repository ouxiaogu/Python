# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-07-03 11:19:46

ImGUI:
    - Image input/ output
    - Image plot utility module

Last Modified by:  ouxiaogu
"""

import numpy as np
import re
import matplotlib.pyplot as plt
import os.path
import sys
import math
import cv2

__all__ = ['readDumpImage', 'readBBox', 'gen_multi_image_overview',
        'imshowCmap', 'cvtFloat2Gray', 'imreadFolder', 'imshowMultiple',
        'imshowMultiple_TitleMatrix', 'read_pgm', 'write_pgm',
        'imread_gray', 'cropToCommonBBox',
        'PolygonDrawer', 'LineDrawer', 'RectangleDrawer',
        'getPolyROI', 'getROIByPointPairs']

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                        dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                        count=int(width)*int(height),
                        offset=len(header)
                        ).reshape((int(height), int(width)))

def write_pgm(img, filename, bbox=None):
    ###write image to a pgm file
    N, M = img.shape;
    maxval = np.iinfo(img.dtype).max

    with open(filename,'wb') as f:
        f.write('P5\n')
        if bbox is None:
            f.write('# bbox: 0 0 {} {}\n'.format(M, N))
        else:
            f.write('# bbox: {} {} {} {}\n'.format(*bbox))
        f.write('{} {}\n'.format(M, N))
        f.write('{}\n'.format(maxval))
        img.tofile(f)

def imread_gray(imfile):
    if not os.path.exists(imfile):
        raise IOError("Error, file doesn't exist at: {}".format(imfile))
    if imfile[-3:] == 'pgm':
        im = cv2.imread(imfile, -1)
        bbox = readBBox(imfile)
    else:
        im = cv2.imread(imfile, 0)
        N, M = im.shape
        bbox = [0, 0, M, N]
    print("bbox in origin image: {}".format(bbox))
    return im, bbox

def readDumpImage(infile, skip_header=0):
    im = np.genfromtxt(infile, skip_header=skip_header).astype('float32')
    imat = np.asmatrix(im)
    return imat

def readBBox(infile):
    bbox = None
    with open(infile, 'rb') as f:
        for i, line in enumerate(f.readlines()):
            if i > 2:
                break
            m = re.search(b"^# bbox:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", line)
            if m is not None:
                bbox = m.groups()
                break
    bbox = list(int(d) for d in bbox)
    return bbox

def cropToCommonBBox(imgs, bboxes):
    xinis, yinis, xends, yends = zip(*bboxes)
    xini, yini, xend, yend = max(xinis), max(yinis), min(xends), min(yends)
    dst = [im[yini:yend, xini:xend] for im in imgs]
    return dst

def gen_multi_image_overview(src_dir, reg_pattern=None, title=None, im_name=None):
    '''generate a overview for multiple image with image title

    command:

    montage -label Balloon   balloon.gif  \
            -label Medical   medical.gif  \
            -tile 4x  -frame 5  -geometry '300x300+2+2>' \
            import os.path
            -title 'My Images'     titled.jpg
    '''
    import os
    from subprocess import call
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
    from FileUtil import FileScanner, getFileLabels
    fsc = FileScanner(src_dir)
    files = fsc.scan_files(regex_pattern=reg_pattern)
    files.sort()
    labels = getFileLabels(files)
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

    sys.stdout.write(command)
    call(command, shell=True)
    sys.stdout.write("please find overview image at {}".format(outfile))

def imreadFolder(src_dir, reg_pattern=None):
    '''
    Read multiple images, read pgm by pgm util, read other by cv

    Parameters
    ----------
    src_dir: str
        image directory
    reg_pattern: regex pattern
        to filter the image files

    Returns
    -------
    images : list
        list of images, image is read in as numpy 2d array
    filelabels: list
        list of image file label, {file} = "{dir}/{label}.{extension}", here
        just need the labels as subplot titles
    '''
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
    from FileUtil import FileScanner, getFileLabels
    fsc = FileScanner(src_dir)
    files = fsc.scan_files(regex_pattern=reg_pattern)
    filelabels = getFileLabels(files)

    images = []
    for imgfile in files:
        if imgfile[-3:0] == 'pgm':
            images.append(read_pgm(imgfile) )
        else:
            images.append(cv2.imread(imgfile, 0) )
    return images, filelabels

def imshowCmap(im, title=None, cmap='gray'):
    """show image in colormap mode, suitable for raw image with negative value"""
    fig, ax = plt.subplots()
    cax = ax.imshow(im, interpolation='none', cmap=cmap)
    fig.colorbar(cax)
    if title is not None:
        ax.set_title(title)
    plt.show()

def cvtFloat2Gray(im):
    """ (x - min)/(max - min)*255 """
    vmin = np.min( im.flatten() )
    vmax = np.max( im.flatten() )
    return np.array((im - vmin)/(vmax - vmin)*255, dtype = np.uint8)


def imshowMultiple(images, titles=None, nrows=None, ncols=4, cmap="gray", axis_on=False, **kwargs):
    '''
    show Multiple Image in one function

    Parameters:
    -----------
    images : array like, container of 2D images
        contains k images, {I1, I2, ..., Ik}, each image size is MxN
    titles : array like, container of string
        each string object is the title of one subplot
    nrows : int
        subplot row number
    ncols : int
        subplot column number, default value is 4
    kwargs : dict like
        pass args like {'vmin': 0, 'vmax': 255} into plt.imshow

    Returns
    -------
    None, directly display multiple images
    '''

    nimg = len(images)
    if titles is None or len(titles) != nimg:
        sys.stderr.write("Warning, input titles is none or invalid, use default naming instead.\n")
        titles = ['image '+str(i) for i in range(nimg)]
    '''
    # comment to support image with different size
    for dimSz in zip(*([im.shape for im in images])): # all images have the same shape
        if( dimSz.count(dimSz[0]) != len(dimSz) ):
            raise ValueError("input images have different size: {}".format(str(dimSz)))
    '''

    if nrows is None or ncols is None or nrows*ncols < nimg:
        # sys.stderr.write("Warning, input nrows/ncols is none or invalid, use build-in smart setting instead\n")
        nrows = math.sqrt(nimg/1.68);
        ncols = nimg/nrows;
        nrows = math.ceil(nrows);
        ncols = math.ceil(ncols);
        if (nrows-1)*ncols >= nimg:
            nrows = nrows - 1;
        elif nrows*(ncols-1) >= nimg:
            ncols = ncols - 1;

    sys.stdout.write("imshowMultiple: {} images, {} x {}.\n".format(nimg, nrows, ncols) )

    # field: areas to plot images; space: areas for margin and subtitles
    # all the w and h variables are normalized into range 0 ~ 1
    field_w_total = 0.9;
    field_h_total = 0.85;
    field_w = field_w_total / ncols;
    field_h = field_h_total / nrows;

    bottom_space_h = 0.01
    space_w = (1 - field_w_total) / (ncols + 1);
    space_h = (1 - field_h_total - bottom_space_h) / nrows;

    fig = plt.figure()
    plt.axis('off')
    for ix in range(nimg):
        # % calculate current row and column of the subplot
        row = ix // ncols;
        col = ix % ncols;

        # % calculate the left, bottom coordinate of this subplot
        field_l = field_w * col + space_w * (col + 1);
        field_b = (nrows - 1 - row) * (field_h + space_h) + bottom_space_h; # axis coord is Y up

        # %  plot the subplot
        ax = fig.add_axes( [field_l, field_b, field_w, field_h] );
        ax.imshow(images[ix], cmap=cmap, **kwargs)
        if not axis_on:
            ax.set_axis_off()
        ax.set_title(titles[ix])

def imshowMultiple_TitleMatrix(images, nrows, ncols, row_titles, col_titles, cmap="gray", x_cmap=None, cbar=False, axis_on=False, **kwargs):
    '''
    similar with `imshowMultiple`, but share the X axis& Y axis title

    Parameters:
    -----------
    images : array like, container of 2D images
        contains k images, {I1, I2, ..., Ik}, each image size is MxN
    nrows : int
        subplot row number
    ncols : int
        subplot column number, default value is 4
    row_titles : array like, container of string
        titles in in row head
    col_titles : array like, container of string
        titles in in column head
    cbar : bool
        whether to show colorbar
    kwargs : dict like
        pass args like {'vmin': 0, 'vmax': 255} into plt.imshow

    Returns
    -------
    None, directly display multiple images
    '''
    nimg = len(images)
    '''
    # comment to support image with different size
    for dimSz in zip(*([im.shape for im in images])): # all images have the same shape
        if( dimSz.count(dimSz[0]) != len(dimSz) ):
            raise ValueError("input images have different size: {}".format(str(dimSz)))
    '''

    if nrows is None or ncols is None or nrows*ncols < nimg:
        raise ValueError("input nrows/ncols is none or invalid: image number: {}, display row x col: {} x {}\n".format(nimg, nrows, ncols) )
    if nrows != len(row_titles):
        raise ValueError("invalid input for row_titles: display rows: {}, #row_titles: {}\n".format(nrows, row_titles) )
    if ncols != len(col_titles):
        raise ValueError("invalid input for col_titles: display cols: {}, #col_titles: {}\n".format(ncols, col_titles) )
    if x_cmap != None:
        assert(len(x_cmap) == ncols)
    sys.stdout.write("imshowMultiple_TitleMatrix: {} images, {} x {}.\n".format(nimg, nrows, ncols) )

    # field: areas to plot images; space: areas for margin and subtitles
    # all the w and h variables are normalized into range 0 ~ 1
    title_w = 0.15
    title_h = 0.1
    field_w_total = 0.75;
    field_h_total = 0.8;
    field_w = field_w_total / ncols;
    field_h = field_h_total / nrows;

    bottom_space_h = 0.01
    space_w = (1 - field_w_total - title_w) / ncols;
    space_h = (1 - field_h_total - title_h - bottom_space_h) / nrows;

    fig = plt.figure()
    # x axis title
    ax = fig.add_axes( [0, 0, 1, 1] );
    y = 1 - title_h*0.9
    for i in range( ncols ):
        x = title_w + (space_w + field_w)*(i + 0.5)
        ax.text(x, y, col_titles[i],
         horizontalalignment='center',
         fontsize=12)
    # y axis title
    x = title_w * 0.1
    for j in range( nrows ):
        y = bottom_space_h + (space_h + field_h)*(nrows - 1 - j + 0.5)
        ax.text(x, y, row_titles[j],
         horizontalalignment='left',
         fontsize=12)
    for ix in range(nimg):
        # % calculate current row and column of the subplot
        row = ix // ncols;
        col = ix % ncols;

        # % calculate the left, bottom coordinate of this subplot
        field_l = title_w + (field_w +  space_w) * col
        field_b = (nrows - 1 - row) * (field_h + space_h) + bottom_space_h; # axis coord is Y up

        # %  plot the subplot
        ax = fig.add_axes( [field_l, field_b, field_w, field_h] );
        cmap = cmap if x_cmap is None else x_cmap[col]
        cax = ax.imshow(images[ix], cmap=cmap, **kwargs)
        if not axis_on:
            ax.set_axis_off()
        if cbar:
            fig.colorbar(cax)

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
    def __init__(self, im, window_name='Drawer'):
        self.raw = im
        if np.ndim(im) == 2:
            self.im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR);
        elif np.ndim(im) == 3:
            self.im = im
        else:
            sys.exit("image need to be an instance of numpy ndarray \n")
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
        self.mode_HV45 = kwargs.get('HV45', True)
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
                lastpair = [self.points[-2], self.points[-1]]
                if self.mode_HV45:
                    lastpair = self._enforce_HV45(lastpair)
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

    def _enforce_HV45(self, src):
        '''
        angle type 0, 1, 2, 3
        - line_type 0: Horizontal line, (x0, y0), (x1, y0)
        - line_type 2: Vertical line, (x0, y0), (x0, y1)
        - line_type 1,3: 45*N line, (x0, y0), (x1, y1')
        - if without enforce line_type: any direction, (x0, y0), (x1, y1)
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
    def __init__(im, window_name):
        kwargs = {'line_type': line_type}
        super(LineDrawer, self).__init__(im, window_name, **kwargs)

class RectangleDrawer(PointPairDrawer):
    """Rectangle is composed by a pair of point: tl and br"""
    def __init__(self, im, window_name):
        kwargs = {'drawfunc': cv2.rectangle}
        super(RectangleDrawer, self).__init__(im, window_name, **kwargs)

if __name__ == '__main__':
    '''test 1'''
    # path = r"C:\Localdata\D\4Development\imageSynthesisTool\data\image"
    path = r"C:\Localdata\D\4Development\imageSynthesisTool\data\orth\avgImage"
    images, labels = imreadFolder(path, reg_pattern='.tif') # not "*" in regex
    # print(labels)
    imshowMultiple(images)
