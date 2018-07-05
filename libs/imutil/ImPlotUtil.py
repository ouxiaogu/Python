"""
-*- coding: utf-8 -*-
Created: peyang, 2018-01-25 11:15:40

Last Modified by: ouxiaogu

ImPlotUtil: plot utility module for Image
"""

import matplotlib.pyplot as plt
import numpy as np

__all__ = ['imshowCmap', 'cvtFloat2Gray']

def imshowCmap(im, title=None, cmap='gray'):
    """show image in colormap mode, suitable for raw image with negative value"""
    fig, ax = plt.subplots()
    cax = ax.imshow(im, interpolation='none', cmap=cmap)
    cbar = fig.colorbar(cax) # ticks=[-1, 0, 1]
    # cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar
    if title is not None:
        ax.set_title(title)
    plt.show()

def cvtFloat2Gray(im):
    """ (x - min)/(max - min)*255 """
    vmin = np.min( im.flatten() )
    vmax = np.max( im.flatten() )
    return np.array((im - vmin)/(vmax - vmin)*255, dtype = np.uint8)