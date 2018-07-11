"""
-*- coding: utf-8 -*-
Created: ouxiaogu, 2018-07-03 11:19:46

ImGUI:
    - Image input/ output
    - Image plot utility module

Last Modified by: ouxiaogu
"""

import numpy as np
import re
import matplotlib.pyplot as plt
import sys
import math
import cv2

__all__ = ['readDumpImage', 'readBBox', 'gen_multi_image_overview',
        'imshowCmap', 'cvtFloat2Gray', 'imreadFolder', 'imshowMultiple',
        'imshowMultiple_TitleMatrix', 'read_pgm', 'write_pgm']

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

def write_pgm(img, filename, byteorder='>'):
    ###write image to a pgm file
    Nx, Ny = img.shape;
    maxval = img.max();

    with open(filename,'wb') as f:
        f.write('P5 {} {} {}\n'.format(Ny, Nx, maxval))
        img.tofile(f)

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
    import os
    from subprocess import call
    sys.path.append("../common")
    from FileUtil import FileScanner, getFileLabel
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
    sys.path.append("../common")
    from FileUtil import FileScanner, getFileLabel
    fsc = FileScanner(src_dir)
    files = fsc.scan_files(regex_pattern=reg_pattern)
    filelabels = getFileLabel(files)

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


def imshowMultiple(images, titles=None, nrows=None, ncols=4, cmap="gray"):
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

    Returns
    -------
    None, directly display multiple images
    '''

    nimg = len(images)
    if titles is None or len(titles) != nimg:
        sys.stderr.write("Warning, input titles is none or invalid, use default naming instead.\n")
        titles = ['image '+str(i) for i in range(nimg)]
    for dimSz in zip(*([im.shape for im in images])): # all images have the same shape
        if( dimSz.count(dimSz[0]) != len(dimSz) ):
            raise ValueError("input images have different size: {}".format(str(dimSz)))

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
        ax.set_axis_off()
        ax.imshow(images[ix], cmap=cmap)
        ax.set_title(titles[ix])

def imshowMultiple_TitleMatrix(images, nrows, ncols, row_titles, col_titles, cmap="gray", x_cmap=None):
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

    Returns
    -------
    None, directly display multiple images
    '''
    nimg = len(images)
    for dimSz in zip(*([im.shape for im in images])): # all images have the same shape
        if( dimSz.count(dimSz[0]) != len(dimSz) ):
            raise ValueError("input images have different size: {}".format(str(dimSz)))

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
        ax.set_axis_off()
        cmap = cmap if x_cmap is None else x_cmap[col]
        cax = ax.imshow(images[ix], cmap=cmap)
        fig.colorbar(cax)

if __name__ == '__main__':
    '''test 1'''
    # path = r"C:\Localdata\D\4Development\imageSynthesisTool\data\image"
    path = r"C:\Localdata\D\4Development\imageSynthesisTool\data\orth\avgImage"
    images, labels = imreadFolder(path, reg_pattern='.tif') # not "*" in regex
    # print(labels)
    imshowMultiple(images)
