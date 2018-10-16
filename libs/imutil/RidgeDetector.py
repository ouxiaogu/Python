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
import cv2

from ImGUI import readBBox
from ImDescriptors import getImageInfo

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../signal")
from filters import cv_gaussian_kernel, gaussian_filter, applySepFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
import logger
log = logger.setup('RidgeDetector', 'debug')

__all__ = ['decideAngleType', 'RidgeDetector']

g_epslmt = 1e-9

DIRS = [(0, 1), (1, 1), (1, 0), (1, -1)]

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

def connectedAdj(b_visited, v_gNL, v_gNH):
    if not b_visited and (v_gNH > 0 or v_gNL > 0):
        return True
    else:
        return False

class RidgeDetector(object):
    """docstring for RidgeDetector"""
    def __init__(self, imfile, sigma=1, thresL=0.1, thresH=0.3, gapLimit=5, minSegLength=10, ksize=None):
        super(RidgeDetector, self).__init__()
        self.imfile = imfile
        self.__readImage()
        self.sigma = sigma
        self.ksize = ksize
        self.thresL = thresL
        self.thresH = thresH
        self.gapLimit = gapLimit
        self.minSegLength = minSegLength

    def __readImage(self):
        if self.imfile[-3:] == 'pgm':
            self.im = cv2.imread(self.imfile, -1)
            bbox = readBBox(self.imfile)
        else:
            self.im = cv2.imread(self.imfile, 0)
            N, M = self.im.shape
            bbox = [0, 0, M, N]
        log.debug("bbox in origin image: {}".format(bbox))
        self.bbox = bbox
        self.height, self.width = self.im.shape

    def computeImageRespones(self):
        flt_G = gaussian_filter(self.sigma)
        flt_dG = gaussian_filter(self.sigma, derivative_order=1)
        flt_ddG = gaussian_filter(self.sigma, derivative_order=2)

        self.Ig = applySepFilter(self.im, flt_G, flt_G)
        self.Ig_dx = applySepFilter(self.im, flt_dG, flt_G)
        self.Ig_dy = applySepFilter(self.im, flt_G, flt_dG)
        self.Ig_dxdx = applySepFilter(self.im, flt_G, flt_ddG)
        self.Ig_dxdy = applySepFilter(self.im, flt_dG, flt_dG)
        self.Ig_dydy = applySepFilter(self.im, flt_ddG, flt_G)

        erode = len(flt_G)//2 + 1 # +1 for 
        self.bbox = [self.bbox[0]+erode, self.bbox[1]+erode, self.bbox[2]-erode, self.bbox[3]-erode]
        log.debug("bbox after computeImageRespones: {}".format(self.bbox))

    def computeImageRidge(self):
        Rg_Mag = np.zeros_like(self.Ig)
        Rg_OrgMag = np.zeros_like(self.Ig)
        Rg_Nx = np.zeros_like(self.Ig)
        Rg_Ny = np.zeros_like(self.Ig)
        Rg_angle = np.zeros_like(self.Ig)

        for i in range(self.bbox[1], self.bbox[3]):
            for j in range(self.bbox[0], self.bbox[2]):
                hessian = np.array( [ self.Ig_dxdx[i, j], self.Ig_dxdy[i, j],
                                    self.Ig_dxdy[i, j], self.Ig_dydy[i, j] ]).reshape((2,2) ) #mxn: 2x2
                if abs(np.linalg.det(hessian)) > g_epslmt:
                    eigen_values, eigen_vectors = np.linalg.eigh(hessian) # mxm, mxn, nxn
                    if i==(self.bbox[1]+self.bbox[3])//2 and j==(self.bbox[0]+self.bbox[2])//2:
                        log.debug("center eigenvalues {}".format(eigen_values))
                        log.debug("center eigenvectors:\n{}".format(eigen_vectors))
                    if eigen_values[0] < eigen_values[1]:
                        eigenIdx = 0
                    else:
                        eigenIdx = 1
                    minEigenVal = eigen_values[eigenIdx] 
                    if minEigenVal < 0: # only negative eigenvalue could be ridge
                        minEigenVal = -eigen_values[eigenIdx]
                        minEigenVec = eigen_vectors[:, eigenIdx] # store positive ridge for nms
                        minAngle = np.arctan2(minEigenVec[1], minEigenVec[0])
                        Rg_Mag[i, j] = minEigenVal
                        Rg_Nx[i, j] = minEigenVec[0]
                        Rg_Ny[i, j] = minEigenVec[1]
                        Rg_angle[i, j] = minAngle
                    Rg_OrgMag[i, j] = abs(eigenvalue[0]) if abs(eigenvalue[0]) > abs(eigenvalue[1]) else abs(eigenvalue[1])

        self.Rg_Mag = Rg_Mag
        self.Rg_Nx = Rg_Nx
        self.Rg_Ny = Rg_Ny
        self.Rg_angle = Rg_angle
        self.Rg_OrgMag = Rg_OrgMag # max abs eigenvalue

    def nonmaxSuppress(self, G, theta):
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
        thetaType = np.zeros(G.shape, int)
        for i in range(self.bbox[1], self.bbox[3]):
            for j in range(self.bbox[0], self.bbox[2]):
                angle_type = decideAngleType(theta[i, j])
                if angle_type in list(range(len(DIRS))):
                    dy, dx = DIRS[angle_type]
                    val = [G[i-dy, j-dx], G[i, j], G[i+dy, j+dx]]
                else:
                    raise ValueError("angle type {} not in {}".format(angle_type, np.arange(len(DELIMITERS))))
                if val[1] >= val[0] and val[1] >= val[-1]:
                    gN[i, j] = G[i, j]
                    thetaType[i, j] = angle_type
        return gN, thetaType

    def doubleThresTraceContour(self):
        self.pointTitle = ["offsetx", "offsety", "angle", "weight", "confidence", "intensity", "slope", "band_width",  "ridge_intensity", "curvature",  "contrast"]
        self.gN = self.Rg_Nms
        # minv = np.min(gN)
        maxv = np.max(self.gN)
        # gNH = self.gN > self.thresH*(maxv-minv) + minv
        # gNL = self.gN > self.thresL*(maxv-minv) + minv
        gNH = self.gN > self.thresH*maxv
        gNL = self.gN > self.thresL*maxv
        gNL = ~gNH&gNL
        self.gNH = gNH
        self.gNL = gNL

        self.imcontour = np.zeros(self.im.shape, bool)
        # imcontour, contour = self.traceContour(gNH, gNL)
        imcontour, contour = self.traceContour2(gNH, gNL)
        contourDetails = []
        for seg in contour:
            if len(seg) > self.minSegLength:
                seg = [(m, n, self.Rg_angle[n, m], 1, 
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
        dirs = [(1,0), (1,1), (0,1), (-1,1), (-1, 0), (-1, -1), (0, -1)]
        for i in range(self.bbox[1], self.bbox[3]):
            for j in range(self.bbox[0], self.bbox[2]):
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
        contour = []
        for i in range(self.bbox[1], self.bbox[3]):
            for j in range(self.bbox[0], self.bbox[2]):
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
        return x >= self.bbox[1] and x < self.bbox[3] and y >= self.bbox[0] and y < self.bbox[2]

    def clearImResponesBorder(self):
        nrows, ncols = self.im.shape
        xx, yy = np.meshgrid(np.arange(ncols), np.arange(nrows))
        mask = ~np.logical_and(np.logical_and(xx >= self.bbox[1], xx < self.bbox[3]), np.logical_and(yy >= self.bbox[0], yy < self.bbox[2]))

        imtypes = ['Ig', 'Ig_dx', 'Ig_dy', 'Ig_dxdx', 'Ig_dxdy', 'Ig_dydy']
        for imtype in imtypes:
            exec('self.{} = self.clearImBorder(self.{}, mask)'.format(imtype, imtype))
            if imtype == 'Ig':
                log.debug("{}: {}".format(imtype, getImageInfo(self.Ig)))

    def cropImagesToBBox(self):
        xini, yini, xend, yend = self.bbox
        imtypes = ['Ig', 'Ig_dx', 'Ig_dy', 'Ig_dxdx', 'Ig_dxdy', 'Ig_dydy', 'Rg_Mag', 'Rg_Nms', 'Rg_OrgMag', 'gN', 'gNL', 'gNH', 'im', 'imcontour']
        for imtype in imtypes:
            exec('self.{} = self.{}[{}:{}, {}:{}]'.format(imtype, imtype, yini, yend, xini, xend))

    def clearImBorder(self, im, mask):
        imat = np.ma.array(im, mask=mask) 
        return imat.filled(0) # fill 0 for masked area, keep unmasked area

    def run(self):
        self.computeImageRespones()

        self.computeImageRidge()
        log.debug("ridge magnitude: {}".format(getImageInfo(self.Rg_Mag)))

        # non-maximum suppression
        self.Rg_Nms, self.Rg_thetaType = self.nonmaxSuppress(self.Rg_Mag, self.Rg_angle)
        log.debug("ridge magnitude non-maximum suppression: {}".format(getImageInfo(self.Rg_Nms)))

        # double thresholding, tracing contour
        self.doubleThresTraceContour()

    def getTotalPointNumber(self):
        totalPoints = 0
        for i, seg in enumerate(self.contour):
            totalPoints += len(seg)
        return totalPoints

    def saveContour(self, contourfile):
        self.contourTitle = "width   height  pixel   centerx centery bbox_xini   bbox_yini   bbox_xend   bbox_yend".split()
        with open(contourfile, 'w+') as fh:
            fh.write('Version v0.1\n')
            fh.write('#Ridge Detection testing\n')
            fh.write('\t'.join(self.contourTitle)+'\n')
            fh.write('\t'.join(self.pointTitle)+'\n')
            fh.write('\n')
            fh.write('%d\t%d\t%f\t%f\t%f\t%s\t%s\t%s\t%s\n' %
                    (self.width, self.height, 1, -1, -1,
                    self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]))

            totalPoints = self.getTotalPointNumber()
            fh.write(str(len(self.contour))+'\t'+ str(totalPoints)+'\n')

            header = self.pointTitle
            formater = ["{}" for i in range(len(header))]
            formater = '\t'.join(formater)
            formater += '\n'
            for i, seg in enumerate(self.contour):
                fh.write(str(len(seg))+'\t'+str(i)+'\n')
                for point in seg:
                    fh.write(formater.format(*point))

'''
void SEMImageRidgeDetector::autoset_thresvalues(PixelType &thresLow, PixelType &thresHigh) {
  m_magHistogram.resize(101);
  for (int iy = Rg_Mag().bbox.yini; iy < Rg_Mag().bbox.yend; iy++) {
    for (int ix = Rg_Mag().bbox.xini; ix < Rg_Mag().bbox.xend; ix++) {
      if (Rg_Mag().pixRef(ix, iy) > 1E-8) {
        int magIdx = static_cast<int>(rint(100 * (Rg_Mag().pixRef(ix, iy)) / maxRgMag()));
        assert(magIdx < 101);
        m_magHistogram[magIdx]++;
      }
    }
  }
  vector<int>::iterator itMax = max_element(m_magHistogram.begin(), m_magHistogram.end());
  double posMax = distance(m_magHistogram.begin(), itMax) / 100.0;

  thresLow = posMax * 3;
  thresHigh = thresLow + 0.3; // TODO: hard coded here. thres high is not studied in details now.
  fprintf(stderr, "autoset thresvalues, thres low is set as %f, thres high is set as %f\n", thresLow, thresHigh);
}
'''