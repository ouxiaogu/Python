class WeightedEdge(Edge):
    """docstring for WeightedEdge"""
    def __init__(self, src, dest, totalDistance, outdoorDistance):
        Edge.__init__(self, src, dest)
        self.totalDistance = totalDistance
        self.outdoorDistance = outdoorDistance
    def getTotalDistance(self):
        return self.totalDistance
    def getOutdoorDistance(self):
        return self.outdoorDistance
    def __str__(self):
        return '{0}->{1} ({2}, {3})'.format(self.src, self.dest, self.totalDistance, self.outdoorDistance)

class WeightedDigraph(Digraph):
    def __init__(self):
        Digraph.__init__(self)
        # self.totalDistance = {}
        # self.outdoorDistance = {}

    def addNode(self, node):
        if node in self.nodes:
            # Even though self.nodes is a Set, we want to do this to make sure we
            # don't add a duplicate entry for the same node in the self.edges list.
            raise ValueError('Duplicate node')
        else:
            self.nodes.add(node)
            self.edges[node] = []
            # self.totalDistance[node] = {}
            # self.outdoorDistance[node] = {}

    def addEdge(self, wtEdge):
        src = wtEdge.getSource()
        dest = wtEdge.getDestination()
        totDist = float(wtEdge.getTotalDistance())
        outDist = float(wtEdge.getOutdoorDistance())
        if not(src in self.nodes and dest in self.nodes):
            raise ValueError('Node not in graph')
        self.edges[src].append([dest, (totDist, outDist)])
        # self.totalDistance[src][dest] = wtEdge.getTotalDistance()
        # self.outdoorDistance[src][dest] = wtEdge.getOutdoorDistance()

    def childrenOf(self,node):
        wtNodes = self.edges[node]
        return [x[0] for x in wtNodes]

    def __str__(self):
        res = ''
        for k in self.edges: # key of the dictionary is node
            for d in self.edges[k]:
                # res = '{}{}->{} ({:.1f}, {:.1f})\n'.format(res, k.name, d.name, self.totalDistance[k][d], self.outdoorDistance[k][d])
                res = '{}{}->{} ({:.1f}, {:.1f})\n'.format(res, k.name, d[0].name, d[1][0], d[1][1])
        return res[:-1]

def bruteForceSearch(digraph, start, end, maxTotalDist, maxDistOutdoors):
    """
    Finds the shortest path from start to end using brute-force approach.
    The total distance travelled on the path must not exceed maxTotalDist, and
    the distance spent outdoor on this path must not exceed maxDistOutdoors.

    Parameters:
        digraph: instance of class Digraph or its subclass
        start, end: start & end building numbers (strings)
        maxTotalDist : maximum total distance on a path (integer)
        maxDistOutdoors: maximum distance spent outdoors on a path (integer)

    Assumes:
        start and end are numbers for existing buildings in graph

    Returns:
        The shortest-path from start to end, represented by
        a list of building numbers (in strings), [n_1, n_2, ..., n_k],
        where there exists an edge from n_i to n_(i+1) in digraph,
        for all 1 <= i < k.

        If there exists no path that satisfies maxTotalDist and
        maxDistOutdoors constraints, then raises a ValueError.
    """
    #TODO

    """
    The flow of this implementation:
        1. find all the possible paths, paths = [ path1, path2, ... ], path1 = ['1', '2', ]
        2. recursively find all weightedPage, wtPaths = [wtPath1, wtPath2, ..  ], wtPath1 = [('1', '2', totalDist, outDist) ]
        3. filter all validWtPaths: maxTotalDist, maxDistOutdoors, choose the shortest sumTotDist in the validWtPath as the winner of this path
    """

    ## https://www.python.org/doc/essays/graphs/
    def findAllPaths(digraph, start, end, path = []):
        # if not ( digraph.hasNode( Node(start) ) and digraph.hasNode( Node(end) ) ):
            # raise KeyError
        # path is used to avoid cycles (the first 'if' inside the 'for' loop).
        # The 'path' argument is not modified: the assignment "path = path + [start]" creates a new list. If we had written "path.append(start)" instead, we would have modified the variable 'path' in the caller, with disastrous results.
        path = path + [start] # make sure the same depth is the same individual unique list
        if start == end :
            return [path] # path embraced by [] means this call of findAllPaths has one and only one valid path
        allpaths = []
        for w in digraph.childrenOf(Node(start)):
            if w.name not in path:
                newpaths = findAllPaths(digraph, w.name, end, path)
                for newpath in newpaths:
                    allpaths.append(newpath)
        return allpaths

    paths = findAllPaths(digraph, start, end, path = [])
#    print "findAllPaths", len(paths), paths[0],  paths[len(paths)/2], paths[len(paths)-1]
#    print paths


    # 2. recursively find all weightedPage
    def findWeightedPaths(digraph, path, visitedWtPaths = [[]]):

        if len(path) < 2:
            return visitedWtPaths

        updatedWtPaths = []
        start = path[0]
        end = path[1]
        for v in digraph.edges[Node(start)]:
            if v[0].name == end:
                # print len(visitedWtPaths), len(visitedWtPaths[0])
                for visitedWtPath in visitedWtPaths:
                    # print (start, end, v[1][0], v[1][1])
                    visitedWtPath = visitedWtPath + [(start, end, v[1][0], v[1][1])]
                    updatedWtPaths.append(visitedWtPath)
                    # updatedWtPaths.append( visitedWtPath.append( (start, end, v[1][0], v[1][1]) ) )
        return findWeightedPaths(digraph, path[1:], updatedWtPaths)

    allWeightedPaths = []
    count = 0
    for path in paths:
        if end not in path:
            count += 1
        weightedPaths = findWeightedPaths(digraph, path, [[]])
        for wtPath in weightedPaths:
            allWeightedPaths.append(wtPath)
#    print "findAllPaths", "{} path don't have end {}".format(count, end)
#    print "findWeightedPaths", len(allWeightedPaths), allWeightedPaths[0],  allWeightedPaths[len(allWeightedPaths)/2], allWeightedPaths[len(allWeightedPaths)-1]

    # 3. filter all validWtPaths:
    count = 0
    minPos = 0
    minDis = 0
    validWtPaths =  []
    for wtPath in allWeightedPaths:
        sumTotDist = 0
        sumOutDist = 0
        for src, dest, totalDist, outDist in wtPath:
            sumTotDist += totalDist
            sumOutDist += outDist
        if sumTotDist <= maxTotalDist and sumOutDist <= maxDistOutdoors:
            validWtPaths.append(wtPath)
            if count == 0:
                minPos = count
                minDis = sumTotDist
            elif minDis > sumTotDist:
                minPos = count
                minDis = sumTotDist
            count += 1

    if len(validWtPaths) == 0:
        raise ValueError
    else:

        shortestWtPath = validWtPaths[minPos]
        shortestpath = [x[0] for x in shortestWtPath]
#        print "shortestWtPath:", shortestWtPath
#        print "shortestpath:", shortestpath
        lenWtPath = len(shortestpath)
        shortestpath.append(shortestWtPath[lenWtPath-1][1])
        return shortestpath