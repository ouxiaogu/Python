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
                # res = '{}{}->{} ({:.1f}, {:.1f})\n'.format(res, k.name, d[0].name, d[1][0], d[1][1])
                res = '{}{}->{} ({}, {})\n'.format(res, k.name, d[0].name, d[1][0], d[1][1])
        return res[:-1]
def bruteForceSearch(digraph, start, end, maxTotalDist, maxDistOutdoors):
    def printPath(path):
        result = ''
        for i in range(len(path)):
            result = result + str(path[i])
            if i != len(path) - 1:
                result = result + '->'
        return result

    def pathDist(graph,path):
        totalDist = 0.0
        outdoorDist = 0.0
        for i in range(len(path) - 1):
            srcEdges = graph.edges[path[i]]
            pathFound = False
            for d in srcEdges:
                if d[0] == path[i+1]:
                    pathFound = True
                    totalDist += d[1][0]
                    outdoorDist += d[1][1]
            if not pathFound:
                return 0.0, 0.0
        return totalDist, outdoorDist

    def DFS(graph, start, end, path, shortest):
        path = path + [start]
        pd1, pd2 = pathDist(graph,path)
        if pd1 > 0.0:
            1# print "Current DFS path:",printPath(path), "Dist:",pathDist(graph,path)
        if start == end:
            return path
        for node in graph.childrenOf(start):
            if node not in path: # avoid cycles
                if shortest == None: # or pathTotalDist < sTotalDist:
                    newPath = DFS(graph, node, end, path, shortest)
                    if newPath != None:
                        pathTotalDist, pathOutsideDist = pathDist(graph,path)
                        sTotalDist = None
                        if shortest != None:
                            sTotalDist, sOutsideDist = pathDist(graph,shortest)
                        # print "current path dist",pathTotalDist, "shortest path dist",sTotalDist
                        shortest = newPath
        return shortest


    totalDist = 0.0
    outdoorDist = 0.0
    path = []

    dfs = DFS(digraph,Node(start),Node(end), [], None)
    totalDist, outdoorDist = pathDist(digraph, dfs)
    # print "dfs, totaldist, outdoordist", dfs, totalDist,outdoorDist
    if totalDist > maxTotalDist:
        raise ValueError("total distance exceeded")
    if outdoorDist > maxDistOutdoors:
        raise ValueError("outdoor distance exceeded")
    strdfs = [str(d) for d in dfs]
    return strdfs