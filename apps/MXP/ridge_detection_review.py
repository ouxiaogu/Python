#!/usr/bin/env python

import numpy as np
import os
import sys
import re
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.path as Path
import matplotlib.patches as Patches
import math

g_epslmt           = 1E-9

class SEMContour:
    def __init__(self):
        self.fileHandle = None
        self.version = 0
        self.headTitle=[]
        self.columnTitle=[]
        self.fieldSize = [0,0]
        self.pixel = 1.0
        self.offset = [0, 0]
        self.bbox = [sys.maxint, sys.maxint, -sys.maxint-1, -sys.maxint-1]
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
            print 'ERROR: Fail to get contour version'
            return False

        lines=self.readlines(1, '\s+')
        self.polygonNum = int(lines[0][0])

        for i in range(self.polygonNum):
            lines = self.readlines(1, '\s+')
            vertexNum = int(lines[0][0])
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

class ContourBBox:
    def __init__(self, xini, yini, xend, yend):
        self.xini = xini
        self.yini = yini
        self.xend = xend
        self.yend = yend

    def contains(self, point):
        if len(point) < 2: return False
        return self.xini - g_epslmt < point[0] < self.xend - g_epslmt and self.yini - g_epslmt < point[1] < self.yend - g_epslmt

def generate_bbox(point, halfLen):
    if len(point) < 2: return None
    return ContourBBox(point[0] - halfLen, point[1] - halfLen, point[0] + halfLen, point[1] + halfLen)

def generate_bbox_for_hotspots(points, halfLen):
    return [ generate_bbox(point, halfLen) for point in points ]

def bboxes_contain(bboxes, point):
    if len(point) < 2: return True    # if contain, remove, so here return True
    ret = False
    for bbox in bboxes:
        if bbox.contains(point):
            ret = True
            break
    return ret

def get_contour_on_grid(contour):
    gridContour = np.zeros(contour.getshape(), dtype=int)
    contourPolygons = contour.getPolygonData()
    for segment in contourPolygons:
        if len(segment['points']) < 1:
            continue
        prevCoord = (segment['points'][0][0], segment['points'][0][1])
        gridContour[int(round(prevCoord[1])), int(round(prevCoord[0]))] = 1
        for ptIdx in range(1, len(segment['points']), 1):
            coord = (segment['points'][ptIdx][0], segment['points'][ptIdx][1])
            gridContour[int(round(coord[1])), int(round(coord[0]))] = 1
            # interp if delta x == 2 or delta y == 2
            if abs(int(round(coord[1])) - int(round(prevCoord[1]))) == 2 or abs(int(round(coord[0])) - int(round(prevCoord[0]))) == 2:
                newCoord = ((prevCoord[0] + coord[0]) / 2, (prevCoord[1] + coord[1]) / 2)
                gridContour[int(round(newCoord[1])), int(round(newCoord[0]))] = 1
            prevCoord = coord

    return gridContour

def obtain_cross_sum(gridContour, contour, coord, checkHlfLen):
    xini  = max(0, int(round(coord[0])) - checkHlfLen)
    yini  = max(0, int(round(coord[1])) - checkHlfLen)
    xend  = min(contour.getshape()[0], int(round(coord[0])) + checkHlfLen)
    yend  = min(contour.getshape()[1], int(round(coord[1])) + checkHlfLen)

    # build a square
    square = []
    for idx in range(xini, xend - 1, 1):
        square.append(gridContour[yini, idx])
    for idx in range(yini, yend - 1, 1):
        square.append(gridContour[idx, xend - 1])
    for idx in range(xend - 1, xini, -1):
        square.append(gridContour[yend - 1, idx])
    for idx in range(yend - 1, yini, -1):
        square.append(gridContour[idx, xini])
    # append the init point to the last
    square.append(gridContour[yini, xini])
    crossSum = 0
    for idx in range(0, len(square) - 1, 1):
        if 1 == square[idx + 1] - square[idx]:
            crossSum = crossSum + 1

    if crossSum > 2 and False:
        sys.stderr.write("%f,%f\t%d\n" % (coord[0], coord[1], crossSum))
        for y in range(yini, yend, 1):
            line = []
            for p in range(xini, xend, 1):
                if 0 == gridContour[y, p]: line.append(".")
                else: line.append("@")
            sys.stderr.write(" ".join(line) + "\n")
        sys.stderr.write("-------------------------------------------------------------------------------------\n")
    return crossSum

def find_cross_points(gridContour, contour, checkHlfLenList):
    contourPolygons = contour.getPolygonData()
    crosspts = []
    for segment in contourPolygons:
        for point in segment['points']:
            coord = (point[0], point[1])
            crossCnt = 0
            for hlfLen in checkHlfLenList:
                crossSum = obtain_cross_sum(gridContour, contour, coord, hlfLen)
                if crossSum > 2: crossCnt = crossCnt + 1

            if crossCnt > 1:
                crosspts.append(coord)

    return crosspts

def remove_points_around_hotspots(contour, bboxes):
    contourPolygons = contour.getPolygonData()
    newPolygons = []
    polygonIdx  = -1
    for segment in contourPolygons:
        newSegment = {}
        newSegment['points'] = []
        for point in segment['points']:
            coord = (point[0], point[1])
            if bboxes_contain(bboxes, coord):
                if len(newSegment['points']) > 0:
                    polygonIdx = polygonIdx + 1
                    newSegment['polygonId'] = polygonIdx
                    newSegment['vertexNum'] = len(newSegment['points'])
                    newSegment['polygon0Hole1'] = 0
                    newPolygons.append(newSegment)
                    newSegment = {}
                    newSegment['points'] = []
                continue
            newSegment['points'].append(copy.copy(point))
        if len(newSegment['points']) > 0:
            polygonIdx = polygonIdx + 1
            newSegment['polygonId'] = polygonIdx
            newSegment['vertexNum'] = len(newSegment['points'])
            newSegment['polygon0Hole1'] = 0
            newPolygons.append(newSegment)
            newSegment = {}

    newContour = SEMContour()
    newContour.setPolygonData(newPolygons)
    newContour.setshape(copy.copy(contour.getshape()))
    newContour.setPixel(contour.getPixel())
    newContour.setVersion(contour.getVersion())
    newContour.setHeadTitle(copy.copy(contour.getHeadTitle()))
    newContour.setColumnTitle(copy.copy(contour.getColumnTitle()))
    newContour.setOffset(copy.copy(contour.getOffset()))
    newContour.setBBox(contour.getBBox())
    return newContour

def plot_contour(contour):
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
    count = 0
    for segment in contourPolygons:
        for point in segment['points']:
            coord = (point[0], point[1])
            vertsx.append(coord[0])
            vertsy.append(height - 1 - coord[1])
            flipCoord = (coord[0], height - 1 - coord[1])
            angle = -point[2] # Y flip, reverse angle
            Rg    = point[8]
            eigenx, eigeny = gen_eigen_vector_points(flipCoord, angle, Rg)
            count += 1
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
    if len(sys.argv) < 2:
        sys.exit("usage: tachyon_python %s [contour_file]" % sys.argv[0])

    inputContourFile  = sys.argv[1]

    contour = SEMContour()
    contour.parseFile(inputContourFile)
    if not contour:
        sys.exit("ERROR: read in contour file fails\n")

    plot_contour(contour)


if "__main__" == __name__:
    main()
