# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-12 11:01:52

Ridge detector

1. Compute image responses: G, Ix, Iy, Ixx, Iyy, Ixy
2. Compute the image ridge(magnitude and angle)
2. nonmaxima suppression on ridge magnitude
3. double thresholding and trace contour by connectivity analysis

Last Modified by:  ouxiaogu
"""

import numpy as np

from ImFeatures import gradient
from ImDescriptors import getImageInfo

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../signal")
from filters import cv_gaussian_kernel, gaussian_filter, applySepFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
import logger
log = logger.setup('RidgeDectector', 'debug')


DELIMITERS = np.linspace(-22.5, 180-22.5, 5)
g_epslmt = 1e-9

DIRS = [(0, 1), (1, 1), (1, 0), (1, -1)]

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
            if angle_type in list(range(len(DIRS))):
                dy, dx = DIRS[angle_type]
                val = [G[i-dy, j-dx], G[i, j], G[i+dy, j+dx]]
            # if angle_type == 0:
            #     val = [G[i, j-1], G[i, j], G[i, j+1]]
            # elif angle_type == 1:
            #     val = [G[i-1, j-1], G[i, j], G[i+1, j+1]]
            # elif angle_type == 2:
            #     val = [G[i-1, j], G[i, j], G[i+1, j]]
            # elif angle_type == 3:
            #     val = [G[i-1, j+1], G[i, j], G[i+1, j-1]]
            else:
                raise ValueError("angle type {} not in {}".format(angle_type, np.arange(len(DELIMITERS))))
            if val[1] >= val[0] and val[1] >= val[-1]:
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

def connectedAdj(b_visited, v_gNL, v_gNH):
    if not b_visited and (v_gNH > 0 or v_gNL > 0):
        return True
    else:
        return False

class RidgeDetector(object):
    """docstring for RidgeDetector"""
    def __init__(self, im, sigma=1, thresL=0.1, thresH=0.3, gapLimit=5, minSegLength=10, ksize=None):
        super(RidgeDetector, self).__init__()
        self.im = im
        self.sigma = sigma
        self.ksize = ksize
        self.thresL = thresL
        self.thresH = thresH
        self.gapLimit = gapLimit
        self.minSegLength = minSegLength

    def computeImageRespones(self):
        flt_G = gaussian_filter(self.sigma)
        flt_dG = gaussian_filter(self.sigma, derivative_order=1)
        flt_ddG = gaussian_filter(self.sigma, derivative_order=2)

        self.Ig = applySepFilter(self.im, flt_G, flt_G)
        self.Ig_dx = applySepFilter(self.im, flt_dG, flt_G)
        self.Ig_dy = applySepFilter(self.im, flt_G, flt_dG)
        self.Ig_dxdx = applySepFilter(self.im, flt_G, flt_ddG)
        self.Ig_dxdy = applySepFilter(self.im, flt_dG, flt_dG)
        self.Ig_dydy = applySepFilter(self.im, flt_G, flt_ddG)

    def computeImageRidge(self):
        Rg_Mag = np.zeros_like(self.Ig)
        Rg_Nx = np.zeros_like(self.Ig)
        Rg_Ny = np.zeros_like(self.Ig)
        Rg_angle = np.zeros_like(self.Ig)

        N, M = Rg_Mag.shape
        for i in range(1, N-1):
            for j in range(1, M-1):
                hessian = np.array( [ self.Ig_dxdx[i, j], self.Ig_dxdy[i, j],
                                    self.Ig_dxdy[i, j], self.Ig_dydy[i, j] ]).reshape((2,2) ) #mxn: 2x2
                if abs(np.linalg.det(hessian)) > g_epslmt:
                    eigen_values, eigen_vectors = np.linalg.eigh(hessian) # mxm, mxn, nxn
                    if i==N//2 and j==M//2:
                        log.debug("center eigenvalues {}".format(eigen_values))
                        log.debug("center eigenvectors:\n{}".format(eigen_vectors))
                    if eigen_values[0] < eigen_values[1]:
                        eigenIdx = 0
                    else:
                        eigenIdx = 1
                    minEigenVal = eigen_values[eigenIdx] 
                    if minEigenVal < 0: # only negative eigenvalue could be ridge
                        minEigenVec = -eigen_vectors[:, eigenIdx] # store positive ridge for nms
                        minAngle = np.arctan2(minEigenVec[1], minEigenVec[0])
                        Rg_Mag[i, j] = minEigenVal
                        Rg_Nx[i, j] = minEigenVec[0]
                        Rg_Ny[i, j] = minEigenVec[1]
                        Rg_angle[i, j] = minAngle

        self.Rg_Mag = Rg_Mag
        self.Rg_Nx = Rg_Nx
        self.Rg_Ny = Rg_Ny
        self.Rg_angle = Rg_angle

    def doubleThresTraceContour(self):
        self.attrs = ["offsetx", "offsety", "angle", "weight", "confidence", "intensity", "slope", "band_width",  "ridge_intensity", "curvature",  "contrast"]
        gN = self.Rg_Nms
        # minv = np.min(gN)
        maxv = np.max(gN)
        # gNH = gN > self.thresH*(maxv-minv) + minv
        # gNL = gN > self.thresL*(maxv-minv) + minv
        gNH = gN > self.thresH*maxv
        gNL = gN > self.thresL*maxv
        gNL = ~gNH&gNL
        self.gNH = gNH
        self.gNL = gNL

        self.imcontour = np.zeros(self.im.shape, bool)
        # imcontour, contour = self.traceContour(gNH, gNL)
        imcontour, contour = self.traceContour2(gNH, gNL)
        contourDetails = []
        for seg in contour:
            if len(seg) > self.minSegLength:
                seg = [(m, n, self.ridge_angle[n, m], 1, 
                    1, self.im[n, m], self.calcRidgeSlope((n, m)), 0, self.Rg_Mag[(n, m)], 0, 0) 
                for n, m in seg]
                contourDetails.append(seg)
            else:
                for point in seg:
                    imcontour[point] = False
        self.imcontour = imcontour
        self.contour = contourDetails

    def calcRidgeSlope(self, pnt):
        angle_type = self.Rg_thetaType[pnt]
        dy, dx = DIRS[angle_type]

        y, x = pnt
        leftpnt = (y-dy, x-dx)
        rightpnt = (y+dy, x+dx)
        slope = 0
        if self.contains(*leftpnt) and self.contains(*rightpnt):
            slope = (self.Rg_Mag[rightpnt] + self.Rg_Mag[leftpnt] - 2*self.Rg_Mag[pnt])/2
        elif self.contains(*leftpnt):
            slope = self.Rg_Mag[pnt] - self.Rg_Mag[leftpnt]
        elif self.contains(*rightpnt):
            slope = self.Rg_Mag[rightpnt] - self.Rg_Mag[pnt]
        return slope

    def traceContour(self, gNH, gNL):
        '''
        contour point info:
        x, y, seed, intensity, gradient mag, gradient angle

        '''
        visited = np.zeros(gNH.shape, bool)

        contour = []
        N, M = gNH.shape
        dirs = [(1,0), (1,1), (0,1), (-1,1), (-1, 0), (-1, -1), (0, -1)]
        for j in range(1,N-1):
            for i in range(1,M-1):
                if visited[j, i] or gNH[j, i]<=0:
                    continue
                seg = [(j, i)]
                visited[j, i] = True
                n, m = j, i
                connected = True
                while connected:
                    connected = False
                    for dx, dy in dirs:
                        if self.contains(n+dx, m+dy) and \
                            connectedAdj(visited[n+dx, m+dy], gNL[n+dx, m+dy], gNH[n+dx, m+dy]):
                            n, m = n+dx, m+dy
                            seg.append((n, m))
                            visited[n, m] = True
                            connected = True
                            break
                contour.append(seg)
        return (visited, contour)

    def traceContour2(self, gNH, gNL):
        self.gmax = gNH.astype(np.float16)*1 + gNL.astype(np.float16)*0.5
        self.vis = np.zeros(self.im.shape, bool)
        self.dx = [1, 0, -1,  0, -1, -1, 1,  1] # axis 0
        self.dy = [0, 1,  0, -1,  1, -1, 1, -1] # axis 1
        N, M = gNH.shape
        contour = []
        for j in range(1, N-1):
            for i in range(1, M-1):
                if self.vis[j, i] or gNH[j, i]<=0:
                    continue
                seg = self.dfs((j, i))
                contour.append(seg)
        return (self.imcontour, contour)

    def dfs(self, origin):
        q = [origin]
        seg = [origin]
        while len(q) > 0:
            s = q.pop()
            self.vis[s] = True
            self.imcontour[s] = True
            for k in range(len(self.dx)): # dirs
                for c in range(1, self.gapLimit): # allowed max contour point gap
                    nx, ny = s[0] + c * self.dx[k], s[1] + c * self.dy[k]
                    if self.contains(nx, ny) and (self.gmax[nx, ny] >= 0.5) and (not self.vis[nx, ny]):
                        seg.append((nx, ny))
                        q.append((nx, ny))
        return seg

    def contains(self, x, y): # x/y: axis 0/1
        return x >= 0 and x < self.im.shape[0] and y >= 0 and y < self.im.shape[1]
    
    def run(self):
        self.computeImageRespones()

        self.computeImageRidge()

        # non-maximum suppression
        gN, thetaType = nonmaxSuppress(self.Rg_Mag, self.Rg_angle)
        self.Rg_Nms = gN
        self.gN = gN
        self.Rg_thetaType = thetaType
        log.debug("gradient non-maximum suppression: {}".format(getImageInfo(gN)))

        # double thresholding, tracing contour
        self.doubleThresTraceContour()