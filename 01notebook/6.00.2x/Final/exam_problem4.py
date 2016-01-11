class Location(object):

    def __init__(self, x, y):
        """x and y are floats"""
        self.x = x
        self.y = y

    def move(self, deltaX, deltaY):
        """deltaX and deltaY are floats"""
        return Location(self.x + deltaX, self.y + deltaY)

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distFrom(self, other):
        ox = other.x
        oy = other.y
        xDist = self.x - ox
        yDist = self.y - oy
        return (xDist**2 + yDist**2)**0.5

    def __str__(self):
        return '<' + str(self.x) + ', ' + str(self.y) + '>'

class FienceLocation(Location):
    def move(self, dx, dy, fenceType, (leftEdge, rightEdge, topEdge, bottomEdge) ):
        x = self.x
        y = self.y
        if fenceType == "SW": # 4-1
            if x+dx > leftEdge and x+dx < rightEdge:
                x += dx
            if  y+dy > bottomEdge and y+dy < topEdge:
                y += dy
        elif fenceType == "SP_wrong": # leftEdge == rightEdge  # 4-2 if step = 5 then, this is not "SP"
            if x+dx < rightEdge and x+dx > leftEdge:
                x += dx
            elif x+dx > rightEdge:
                x = leftEdge
            elif x+dx < leftEdge:
                x =rightEdge
            if y+dy < topEdge and  y+dy > bottomEdge:
                y += dy
            elif y+dy > topEdge:
                y = topEdge
            elif y+dy < bottomEdge:
                y = bottomEdge
        elif fenceType == "SP": # leftEdge == rightEdge # 4-3
            if x+dx > leftEdge and x+dx < rightEdge:
                x += dx
            elif x+dx > rightEdge:
                x = leftEdge + (x+dx - rightEdge)
            elif x+dx < leftEdge:
                x = rightEdge - (leftEdge - (x+dx))

            if  y+dy > bottomEdge and y+dy < topEdge:
                y += dy
            elif y+dy > topEdge:
                y = bottomEdge + (y+dy - topEdge)
            elif y+dy < bottomEdge:
                y = topEdge - (bottomEdge - (y+dy))
        elif fenceType == "inverse-WW": # 4-4
            if x+dx < rightEdge and x+dx > leftEdge:
                x += dx
            elif x+dx > rightEdge:
                x = bottomEdge
            elif x+dx < leftEdge:
                x = topEdge
            if y+dy < topEdge and  y+dy > bottomEdge:
                y += dy
            elif y+dy > topEdge:
                y = leftEdge
            elif y+dy < bottomEdge:
                y = rightEdge
        elif fenceType == "BackToDiagonal": # 4-5
            if x+dx > rightEdge:
                x = y
            if x+dx < leftEdge:
                x = y
            if x+dx < rightEdge and x+dx > leftEdge:
                x += dx
            if y+dy > topEdge:
                y = x
            if y+dy < bottomEdge:
                y = x
            if y+dy < topEdge and  y+dy > bottomEdge:
                y += dy
        elif fenceType == "BH-wrong, h-width h-height": # 4-6
            if x+dx > rightEdge:
                x,y = (rightEdge-leftEdge)/2
            if x+dx < leftEdge:
                x,y = (rightEdge-leftEdge)/2
            if x+dx < rightEdge and x+dx > leftEdge:
                x += dx
            if y+dy > topEdge:
                x,y = (rightEdge-leftEdge)/2
            if y+dy < bottomEdge:
                x,y = (rightEdge-leftEdge)/2
            if y+dy < topEdge and  y+dy > bottomEdge:
                y += dy
        elif fenceType == "BH": # 4-7
            if x+dx < rightEdge and x+dx > leftEdge and y+dy < topEdge and y+dy > bottomEdge:
                x += dx
                y += dy
            else:
                x = leftEdge + (rightEdge-leftEdge)/2
                y = bottomEdge + (topEdge-bottomEdge)/2
        elif fenceType == "WW": # addition
            if x+dx < rightEdge and x+dx > leftEdge:
                x += dx
            elif x+dx > rightEdge:
                x = topEdge
            elif x+dx < leftEdge:
                x = bottomEdge
            if y+dy < topEdge and  y+dy > bottomEdge:
                y += dy
            elif y+dy > topEdge:
                y = rightEdge
            elif y+dy < bottomEdge:
                y = leftEdge
        return Location(x, y)

class Field(object):

    def __init__(self):
        self.drunks = {}

    def addDrunk(self, drunk, loc):
        if drunk in self.drunks:
            raise ValueError('Duplicate drunk')
        else:
            self.drunks[drunk] = loc

    def moveDrunk(self, drunk):
        if not drunk in self.drunks:
            raise ValueError('Drunk not in field')
        xDist, yDist = drunk.takeStep()
        currentLocation = self.drunks[drunk]
        #use move method of Location to get new location
        self.drunks[drunk] = currentLocation.move(xDist, yDist)
        # self.drunks[drunk] = currentLocation.move(xDist, yDist, "WW")

    def getLoc(self, drunk):
        if not drunk in self.drunks:
            raise ValueError('Drunk not in field')
        return self.drunks[drunk]
class FienceField(Field):
    def __init__(self, (leftEdge, rightEdge, topEdge, bottomEdge) ):
        self.drunks = {}
        self.edges = (leftEdge, rightEdge, topEdge, bottomEdge)
    def moveDrunk(self, drunk, fenceType):
        if not drunk in self.drunks:
            raise ValueError('Drunk not in field')
        xDist, yDist = drunk.takeStep()
        currentLocation = self.drunks[drunk]
        print(type(currentLocation))
        #use move method of Location to get new location
        self.drunks[drunk] = currentLocation.move(xDist, yDist, fenceType, self.edges)

import random


class Drunk(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return 'This drunk is named ' + self.name

class UsualDrunk(Drunk):
    def takeStep(self):
        stepChoices =\
            [(0.0,1.0), (0.0,-1.0), (1.0, 0.0), (-1.0, 0.0)]
        return random.choice(stepChoices)
class DUDrunk(Drunk):
    def takeStep(self):
        stepChoices = random.choice([(-1.0,-1.0),(-1.0,0.0),(-1.0,1.0),(0.0,1.0),(0.0,-1.0),(1.0,-1.0),(1.0,0.0),(1.0,1.0)])
        return stepChoices

def runSim(numSim, fenceType):
    drunkA = DUDrunk("A")
    fienceField = FienceField( (-50, 50, -50, 50) )
    fienceField.addDrunk(drunkA, FienceLocation(0, 0))
    xlocs = []
    ylocs = []
    for i in range(numSim):
        fienceField.moveDrunk(drunkA, fenceType)
        loc = fienceField.getLoc(drunkA)
        xlocs.append(loc.x)
        ylocs.append(loc.y)
    return (xlocs, ylocs)

import pylab
def drunkTestP():
    numSim = 200
    figNo = 0
    for dClass in ("SW", "SP", "WW", "BH"):
        #for numSteps in stepsTaken:
            # distances = simWalks(numSteps, numTrials, dClass)
            # meanDistances.append(sum(distances)/len(distances))
        x, y = runSim(numSim, dClass)
        figNo = figNo + 1
        fig = pylab.figure(figNo)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], label="ax")
        ax.scatter(x, y, color='#990000', #Wine
                   label = dClass.__name__)
        pylab.title('drunk vector distribution')
        pylab.xlabel('x')
        pylab.ylabel('y')
        ax.set_xlim(xmin=-60, xmax=60)
        ax.set_ylim(ymin=-60, ymax=60)
        pylab.legend(loc = 'upper left')

drunkTestP()