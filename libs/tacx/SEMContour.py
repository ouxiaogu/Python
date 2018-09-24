# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-09-19 12:07:08

SEM Contour class definition and IO files

Last Modified by: ouxiaogu
"""

import math
import sys
import re
import numpy as np
import pandas as pd

g_epslmt  = 1E-9

__all__ = ['SEMContour', 'plot_contour']

class SEMContour:
    def __init__(self):
        self.fileHandle = None
        self.version = 0
        self.headTitle=[]
        self.columnTitle=[]
        self.fieldSize = [0,0]
        self.pixel = 1.0
        self.offset = [0, 0]
        self.bbox = [-1, -1, -1, -1]
        self.polygonNum = 0
        self.polygonData = []

    def copy(self):
        return

    def getshape(self):
        return (self.fieldSize[0], self.fieldSize[1])

    def setshape(self, shape):
        self.fieldSize = shape

    def getPolygonData(self):
        return self.polygonData

    def setPolygonData(self, polygons):
        self.polygonData = polygons
        self.polygonNum  = len(polygons)

    def getPixel(self):
        return self.pixel

    def setPixel(self, pixel):
        self.pixel = pixel

    def getVersion(self):
        return self.version

    def setVersion(self, version):
        self.version = version

    def setHeadTitle(self, headTitle):
        self.headTitle = headTitle

    def getHeadTitle(self):
        return self.headTitle

    def setColumnTitle(self, columnTitle):
        self.columnTitle = columnTitle

    def getColumnTitle(self):
        return self.columnTitle

    def setOffset(self, offset):
        self.offset = offset

    def getOffset(self):
        return self.offset

    def setBBox(self, bbox):
        self.bbox = bbox

    def getBBox(self):
        return self.bbox

    def readlines(self, number, delimiter):
        lines = []
        i=0
        while i<number:
            c = self.fileHandle.readline().strip()
            if(len(c)==0):
                continue
            if(c[0]=='#'):
                continue
            fields = re.split(delimiter, c)
            if(len(fields)==0):
                continue
            lines.append([])
            lines[i].extend(fields)
            i=i+1
        return lines

    def parseFile(self, filename):
        self.fileHandle = open(filename, 'r')
        # Read first line: field size
        lines = self.readlines(1, '\s+')
        if(re.match(r'Version', lines[0][0])):
            # New version from E8.0 release.
            m = re.match(r'v(\d+\.?\d+)', lines[0][1])
            if(m):
                self.version = float(m.group(1))
            self.headTitle = self.readlines(1, '[\s\t]+')[0]
            self.columnTitle = self.readlines(1, '[\s\t]+')[0]
            head = self.readlines(1, '\s+')
            self.fieldSize[0] = float(head[0][0])
            self.fieldSize[1] = float(head[0][1])
            self.pixel = float(head[0][2])
            self.offset[0] = float(head[0][3])
            self.offset[1] = float(head[0][4])
            self.bbox = [head[0][5], head[0][6], head[0][7], head[0][8]]
        else:
            print ('ERROR: Fail to get contour version')
            return False

        lines=self.readlines(1, '\s+')
        self.polygonNum = int(lines[0][0])

        for i in range(self.polygonNum):
            lines = self.readlines(1, '\s+')
            vertexNum = int(lines[0][0])
            polygonId = int(lines[0][1])
            if(vertexNum<=0):
                print ("ERROR: PolygonId=",polygonId," has vertex number smaller than 0.")
                return -1
            self.polygonData.append([])
            self.polygonData[i] = {}
            self.polygonData[i]['vertexNum']=vertexNum
            self.polygonData[i]['polygonId']=int(lines[0][1])
            self.polygonData[i]['polygon0Hole1']=0
            self.polygonData[i]['points'] = []

            lines = self.readlines(vertexNum, '\s+')
            for m in range(len(lines)):
                for n in range(len(lines[m])):
                    lines[m][n] = float(lines[m][n])
            self.polygonData[i]['points'].extend(lines)

        self.fileHandle.close()
        return True

    def saveContour(self, filepath):
        fh = open(filepath, 'w')
        fh.write('Version v'+str(self.version)+'\n')
        fh.write('\t'.join(self.headTitle)+'\n')
        fh.write('\t'.join(self.columnTitle)+'\n')
        fh.write('\n')
        fh.write('%d\t%d\t%f\t%f\t%f\t%s\t%s\t%s\t%s\n' %
                (self.fieldSize[0], self.fieldSize[1],
            self.pixel, self.offset[0], self.offset[1],
            self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]))
        totalPoints = 0
        for i in range(len(self.polygonData)):
            totalPoints = totalPoints + self.polygonData[i]['vertexNum']
        fh.write(str(self.polygonNum)+'\t'+ str(totalPoints)+'\n')
        for i in range(len(self.polygonData)):
            fh.write(str(int(self.polygonData[i]['vertexNum']))+'\t'+
                    str(int(self.polygonData[i]['polygonId']))+'\n')
            for j in range(len(self.polygonData[i]['points'])):
                p = self.polygonData[i]['points'][j]
                s = []
                for k in p:
                    s.append(str(k))
                fh.write('\t'.join(s)+'\n')
        fh.close()

    def toDf(self):
        header = ['polygonId']
        header.extend(self.columnTitle)
        allpoints = []
        for polygon in self.polygonData:
            polygonId = polygon['polygonId']
            for point in polygon['points']:
                p = [polygonId]
                p.extend(point)
                allpoints.append(p)
        shape = (len(allpoints), len(header))
        allpoints = np.array(allpoints).reshape(shape)
        df = pd.DataFrame(allpoints, columns=header)
        return df

class ContourBBox:
    def __init__(self, xini, yini, xend, yend):
        self.xini = xini
        self.yini = yini
        self.xend = xend
        self.yend = yend

    def contains(self, point):
        if len(point) < 2: return False
        return self.xini - g_epslmt < point[0] < self.xend - g_epslmt and self.yini - g_epslmt < point[1] < self.yend - g_epslmt

def plot_contour(contour, wi_eigen=False):
    import matplotlib.pyplot as plt
    width  = contour.getshape()[0]
    height = contour.getshape()[1]

    plt.ioff()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    vertsx = []
    vertsy = []
    contourPolygons = contour.getPolygonData()
    for segment in contourPolygons:
        for point in segment['points']:
            coord = (point[0], point[1])
            vertsx.append(coord[0])
            vertsy.append(height - 1 - coord[1])
            if wi_eigen:
                flipCoord = (coord[0], height - 1 - coord[1])
                angle = -point[2] # Y flip, reverse angle
                Rg    = point[8]
                eigenx, eigeny = gen_eigen_vector_points(flipCoord, angle, Rg)
                ax.plot(eigenx, eigeny, 'r-')

    ax.plot(vertsx, vertsy, 'b.', markersize=3)
    plt.show()

def gen_eigen_vector_points(coord, angle, Rg):
    eigenx, eigeny = [], []
    x0, y0 = coord
    vecLen = min(int(200.*Rg + 0.5), 10)
    fVecRange = map(lambda x: x/5., range(-vecLen*5, vecLen*5+1) )
    for dz in fVecRange:
        x = x0 + dz * math.cos(angle)
        y = y0 + dz * math.sin(angle)
        eigenx.append(x)
        eigeny.append(y)
    return (eigenx, eigeny)

def main():
    inputContourFile  = r'C:\Localdata\D\Note\Python\misc\SEM\samples\contour.txt'
    #inputContourFile  = r'C:\Localdata\D\Note\Python\libs\imutil\unittest\contour.txt'

    contour = SEMContour()
    contour.parseFile(inputContourFile)
    if not contour:
        sys.exit("ERROR: read in contour file fails\n")

    plot_contour(contour)


if "__main__" == __name__:
    main()