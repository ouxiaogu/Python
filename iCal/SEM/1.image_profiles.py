# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:05:39 2016

@author: peyang
"""

import numpy as np
import cv2
import os, os.path
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\Localdata\D\Note\Python\FEM\util')
from plot_util import setAxisLim

sempath = os.path.join(os.getcwd(), 'samples')
sem_hv = cv2.imread(os.path.join(sempath, 'HV.tif'))
sem_hv = cv2.cvtColor(sem_hv, cv2.COLOR_RGB2GRAY)
sem_hh = cv2.imread(os.path.join(sempath, 'HH.tif'))
sem_hh = cv2.cvtColor(sem_hh, cv2.COLOR_RGB2GRAY)
#cv2.imshow("hh", sem_hh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

im1_h, im1_w = sem_hv.shape[:2]
im2_h, im2_w = sem_hh.shape[:2]
center = (im1_h/2, im1_w/2)

# rotate the image by 180 degrees
M = cv2.getRotationMatrix2D(center, 90, 1.0)
sem_hh = cv2.warpAffine(sem_hh, M, (im2_w, im2_h))
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8], label="ax1")

ax1.plot(np.average(sem_hv, axis=0), label="sem_hv") # average column by column
# ax1.plot(np.average(sem_hh, axis=1), label="sem_hh") # average row by row, don't need when using sem_hh is rotated 90 clockwise
ax1.plot(np.average(sem_hh, axis=0), label="sem_hh") # average column by column
ax1.set_xlabel("location")
ax1.set_ylabel("SEM intensity")
ax1.set_title("The Profile at feature X direction")
setAxisLim(ax1, 0)
ax1.legend(loc='best')