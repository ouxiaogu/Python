# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-06 20:24:52

Edge detector

classic canny edge detector

1. Smooth input image by Gaussian Filter
2. Compute the image gradient(magnitude and angle)
3. nonmaxima suppression on gradient magnitude
4. double thresholding and trace contour by connectivity analysis

Last Modified by: ouxiaogu
"""
import numpy as np

from ImFeatures import gradient

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../signal")
from filters import cv_gaussian_kernel, gaussian_filter, applySepFilter

DELIMITERS = np.linspace(-22.5, 180-22.5, 5)
g_epslmt = 1e-9

def nonmaxSuppress(G, theta):
    '''
    1. classify the theta into H, 45, V, 135 4 directions
    2. G(p)=0 if G(p-1)>G(p), or G(p+1)>G(p)
    
    Parameters
    ----------
    G, theta : ndarray-like image, ndarray-like image
        gradient magnitude & angle
    
    Returns
    -------
    gN: ndarray-like image
        gradient magnitude applied non-max suppression
    '''
    gN = np.zeros_like(G)
    thetaType = np.zeros_like(theta)
    N, M = G.shape
    for i in range(1,N-1):
        for j in range(1,M-1):
            angle_type = decideAngleType(theta[i, j])
            if angle_type == 0:
                val = [G[i, j-1], G[i, j], G[i, j+1]]
            elif angle_type == 1:
                val = [G[i-1, j-1], G[i, j], G[i+1, j+1]]
            elif angle_type == 2:
                val = [G[i-1, j], G[i, j], G[i+1, j]]
            elif angle_type == 3:
                val = [G[i-1, j+1], G[i, j], G[i+1, j-1]]
            else:
                raise ValueError("angle type {} not in {}".format(angle_type, np.arange(len(DELIMITERS))))
            if val[1] >= val[0] or val[1] >= val[-1]:
                gN[i, j] = G[i, j]
                thetaType[i, j] = angle_type
    return gN, thetaType

def decideAngleType(angle):
    ntypes = len(DELIMITERS)
    ret = 0
    for k in range(ntypes):
        if k == 0:
            lim = [0, DELIMITERS[k+1]]
        elif k < ntypes-1:
            lim = [DELIMITERS[k], DELIMITERS[k+1]]
        elif k == ntypes - 1:
            lim = [DELIMITERS[k], 180]
        if lim[0] <= abs(angle) and abs(angle) < lim[1]:
            ret = k
            break
    ret = ret%(ntypes - 1)
    return ret

class ContourBBox:
    def __init__(self, xini, yini, xend, yend):
        self.xini = xini
        self.yini = yini
        self.xend = xend
        self.yend = yend

    def contains(self, point): #half closed
        if len(point) < 2: return False
        return self.xini - g_epslmt < point[0] < self.xend - g_epslmt and self.yini - g_epslmt < point[1] < self.yend - g_epslmt

class EdgeDetector(object):
    """docstring for EdgeDetector"""
    def __init__(self, im, sigma=0.8, ksize=5, thresL=0.1, thresH=0.3):
        super(EdgeDetector, self).__init__()
        self.im = im
        self.sigma = sigma
        self.ksize = ksize
        self.thresL = thresL
        self.thresH = thresH
        

    def run(self):
        # smooth
        # flt_G = gaussian_filter(self.sigma)
        flt_G = cv_gaussian_kernel(self.ksize, self.sigma)
        gim = applySepFilter(self.im, flt_G, flt_G)
        self.gim = gim

        # gradient
        G, theta = gradient(gim)
        self.G = G

        # nonmaxima suppression
        gN, thetaType = nonmaxSuppress(G, theta)
        self.gN = gN
        self.theta = theta

        # double thresholding, tracing contour
        gcontour, contour = self.doubleThres()
        self.gcontour = gcontour
        self.contour = contour

    def doubleThres(self):
        minv = np.min(self.gN)
        maxv = np.max(self.gN)
        gNH = self.gN > self.thresH*(maxv-minv) + minv
        gNL = self.gN > self.thresL*(maxv-minv) + minv
        gNL = ~gNH&gNL
        print(gNH.shape)
        return self.traceContour(gNH, gNL)
        

    def traceContour(self, gNH, gNL):
        '''
        contour point info:
        x, y, seed, intensity, gradient mag, gradient angle

        '''
        visited = np.zeros(gNH.shape)
        gcontour = np.zeros_like(gNH)

        contour = []
        N, M = gNH.shape
        dirs = [(1,0), (1,1), (0,1), (-1,1), (-1, 0), (-1, -1), (0, -1)]
        bbox = ContourBBox(1, 1, M-1, N-1)
        for j in range(1,N-1):
            for i in range(1,M-1):
                if visited[j, i] or gNH[j, i]<=0:
                    continue
                seg = [(i, j, 1, self.im[j, i], self.gN[j, i], self.theta[j, i])]
                gcontour[j, i] = gNH[j, i]
                n, m = j, i
                connected = True
                while connected:
                    connected = False
                    for dx, dy in dirs:
                        if bbox.contains([m+dx, n+dy]) and \
                            connectedAdj(visited[n+dy, m+dx], gNL[n+dy, m+dx], gNH[n+dy, m+dx], bbox):
                            n, m = n+dy, m+dx
                            seg.append((m, n, 0, self.im[n, m], self.gN[n, m], self.theta[n, m]))
                            gcontour[n, m] = gNH[n, m]
                            visited[n, m] = True
                            connected = True
                            break
                contour.append(seg)
        return (gcontour, contour)

def connectedAdj(b_visited, v_gNL, v_gNH, bbox):
    if not b_visited and (v_gNH != 0 or v_gNH != 0):
        return True
    else:
        return False