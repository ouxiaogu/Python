# 6.00.2x Problem Set 5
# Graph optimization
# Finding shortest paths through MIT buildings
#

import string
# This imports everything from `graph.py` as if it was defined in this file!
from graph import *

#
# Problem 2: Building up the Campus Map
#
# Before you write any code, write a couple of sentences here
# describing how you will model this problem as a graph.

# This is a helpful exercise to help you organize your
# thoughts before you tackle a big design problem!
#

def load_map(mapFilename):
    """
    Parses the map file and constructs a directed graph

    Parameters:
        mapFilename : name of the map file

    Assumes:
        Each entry in the map file consists of the following four positive
        integers, separated by a blank space:
            From To TotalDistance DistanceOutdoors
        e.g.
            32 76 54 23
        This entry would become an edge from 32 to 76.

    Returns:
        a directed graph representing the map
    """
    # TODO
    print "Loading map from file..."
    g = WeightedDigraph()
    with open(mapFilename, 'r+') as infile:
        nodes = []
        lines = infile.readlines()
        for line in lines:
            integers = line.split()
            for i in range(2):
                node = Node(integers[i])
                if node not in nodes:
                    nodes.append(node)
                    g.addNode(node)
        for line in lines:
            integers = line.split()
            wtEdge = WeightedEdge(Node(integers[0]), Node(integers[1]), int(integers[2]), int(integers[3]))
            g.addEdge(wtEdge)
    return g
# Test Problem2
#mitMap = load_map("mit_map.txt")
#print isinstance(mitMap, Digraph)
#print isinstance(mitMap, WeightedDigraph)
#
#print mitMap.nodes
#print mitMap.edges
#print g

#
# Problem 3: Finding the Shortest Path using Brute Force Search
#
# State the optimization problem as a function to minimize
# and what the constraints are
#

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

# https://en.wikipedia.org/wiki/Depth-first_search
# 1  procedure DFS-iterative(G,v):
# 2      let S be a stack
# 3      S.push(v)
# 4      while S is not empty
# 5            v = S.pop()
# 6            if v is not labeled as discovered:
# 7                label v as discovered
# 8                for all edges from v to w in G.adjacentEdges(v) do
# 9                    S.push(w)


#
# Problem 4: Finding the Shorest Path using Optimized Search Method
#
def directedDFS(digraph, start, end, maxTotalDist, maxDistOutdoors):
    """
    Finds the shortest path from start to end using directed depth-first.
    search approach. The total distance travelled on the path must not
    exceed maxTotalDist, and the distance spent outdoor on this path must
    not exceed maxDistOutdoors.

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
    def getDists(digraph, From, To):
        for v in digraph.edges[Node(From)]:
            if v[0].name == To:
                return (v[1][0], v[1][1])
        return (maxTotalDist, maxDistOutdoors)

    def getPathDistance(path):
        distance, outDistance = 0, 0
        if len(path) < 2:
            return distance, outDistance
        for i in range(len(path)-1):
            From, To = path[i], path[i+1]
            totalDist, outDist = getDists(digraph, From, To)
            distance += totalDist
            outDistance += outDist
        return distance, outDistance

    ## https://www.python.org/doc/essays/graphs/
    def findAllPaths2(digraph, start, end, path = []):
        # if not ( digraph.hasNode( Node(start) ) and digraph.hasNode( Node(end) ) ):
            # raise KeyError
        # path is used to avoid cycles (the first 'if' inside the 'for' loop).
        # The 'path' argument is not modified: the assignment "path = path + [start]" creates a new list. If we had written "path.append(start)" instead, we would have modified the variable 'path' in the caller, with disastrous results.
        path = path + [start] # make sure the same depth is the same individual unique list
        sumTotDist, sumOutDist = getPathDistance(path)        
        if start == end:
            if sumTotDist <= maxTotalDist and sumOutDist <= maxDistOutdoors:
                return [path] # path embraced by [] means this call of findAllPaths has one and only one valid path
            else:
                return []
        allpaths = []
        for w in digraph.childrenOf(Node(start)):
            if w.name not in path:
                if sumTotDist <= maxTotalDist and sumOutDist <= maxDistOutdoors:
                    newpaths = findAllPaths2(digraph, w.name, end, path)
                    for newpath in newpaths:
                        allpaths.append(newpath)
        return allpaths


    paths = findAllPaths2(digraph, start, end, [])
    print "findAllPaths2", len(paths) , paths[0],  paths[len(paths)/2], paths[len(paths)-1]

    # 3. filter all validWtPaths:
    if len(paths) == 0:
        raise ValueError
    else:
        shortestpath = paths[0]
        minDis = maxTotalDist
        for path in paths:
            sumTotDist, sumOutDist= getPathDistance(path)
            if minDis > sumTotDist:
                minDis = sumTotDist
                shortestpath = path
        return shortestpath

# Uncomment below when ready to test
#### NOTE! These tests may take a few minutes to run!! ####
if __name__ == '__main__':
#     Test cases
    mitMap = load_map("mit_map.txt")
    # print isinstance(mitMap, Digraph)
    # print isinstance(mitMap, WeightedDigraph)
    # print 'nodes', mitMap.nodes
    # print 'edges', mitMap.edges


    LARGE_DIST = 1000000

#     Test case 1
    print "---------------"
    print "Test case 1:"
    print "Find the shortest-path from Building 32 to 56"
    expectedPath1 = ['32', '56']
    brutePath1 = bruteForceSearch(mitMap, '32', '56', LARGE_DIST, LARGE_DIST)
    dfsPath1 = directedDFS(mitMap, '32', '56', LARGE_DIST, LARGE_DIST)
    print "Expected: ", expectedPath1
    print "Brute-force: ", brutePath1
    print "DFS: ", dfsPath1
    print "Correct? BFS: {0}; DFS: {1}".format(expectedPath1 == brutePath1, expectedPath1 == dfsPath1)

#     Test case 2
    print "---------------"
    print "Test case 2:"
    print "Find the shortest-path from Building 32 to 56 without going outdoors"
    expectedPath2 = ['32', '36', '26', '16', '56']
    brutePath2 = bruteForceSearch(mitMap, '32', '56', LARGE_DIST, 0)
    dfsPath2 = directedDFS(mitMap, '32', '56', LARGE_DIST, 0)
    print "Expected: ", expectedPath2
    print "Brute-force: ", brutePath2
    print "DFS: ", dfsPath2
    print "Correct? BFS: {0}; DFS: {1}".format(expectedPath2 == brutePath2, expectedPath2 == dfsPath2)

#     Test case 3
    print "---------------"
    print "Test case 3:"
    print "Find the shortest-path from Building 2 to 9"
    expectedPath3 = ['2', '3', '7', '9']
    brutePath3 = bruteForceSearch(mitMap, '2', '9', LARGE_DIST, LARGE_DIST)
    dfsPath3 = directedDFS(mitMap, '2', '9', LARGE_DIST, LARGE_DIST)
    print "Expected: ", expectedPath3
    print "Brute-force: ", brutePath3
    print "DFS: ", dfsPath3
    print "Correct? BFS: {0}; DFS: {1}".format(expectedPath3 == brutePath3, expectedPath3 == dfsPath3)

# #     Test case 4
#     print "---------------"
#     print "Test case 4:"
#     print "Find the shortest-path from Building 2 to 9 without going outdoors"
#     expectedPath4 = ['2', '4', '10', '13', '9']
#     brutePath4 = bruteForceSearch(mitMap, '2', '9', LARGE_DIST, 0)
#     dfsPath4 = directedDFS(mitMap, '2', '9', LARGE_DIST, 0)
#     print "Expected: ", expectedPath4
#     print "Brute-force: ", brutePath4
#     print "DFS: ", dfsPath4
#     print "Correct? BFS: {0}; DFS: {1}".format(expectedPath4 == brutePath4, expectedPath4 == dfsPath4)

# #     Test case 5
#     print "---------------"
#     print "Test case 5:"
#     print "Find the shortest-path from Building 1 to 32"
#     expectedPath5 = ['1', '4', '12', '32']
#     brutePath5 = bruteForceSearch(mitMap, '1', '32', LARGE_DIST, LARGE_DIST)
#     dfsPath5 = directedDFS(mitMap, '1', '32', LARGE_DIST, LARGE_DIST)
#     print "Expected: ", expectedPath5
#     print "Brute-force: ", brutePath5
#     print "DFS: ", dfsPath5
#     print "Correct? BFS: {0}; DFS: {1}".format(expectedPath5 == brutePath5, expectedPath5 == dfsPath5)

# #     Test case 6
#     print "---------------"
#     print "Test case 6:"
#     print "Find the shortest-path from Building 1 to 32 without going outdoors"
#     expectedPath6 = ['1', '3', '10', '4', '12', '24', '34', '36', '32']
#     brutePath6 = bruteForceSearch(mitMap, '1', '32', LARGE_DIST, 0)
#     dfsPath6 = directedDFS(mitMap, '1', '32', LARGE_DIST, 0)
#     print "Expected: ", expectedPath6
#     print "Brute-force: ", brutePath6
#     print "DFS: ", dfsPath6
#     print "Correct? BFS: {0}; DFS: {1}".format(expectedPath6 == brutePath6, expectedPath6 == dfsPath6)

# #     Test case 7
#     print "---------------"
#     print "Test case 7:"
#     print "Find the shortest-path from Building 8 to 50 without going outdoors"
#     bruteRaisedErr = 'No'
#     dfsRaisedErr = 'No'
#     try:
#         bruteForceSearch(mitMap, '8', '50', LARGE_DIST, 0)
#     except ValueError:
#         bruteRaisedErr = 'Yes'

#     try:
#         directedDFS(mitMap, '8', '50', LARGE_DIST, 0)
#     except ValueError:
#         dfsRaisedErr = 'Yes'

#     print "Expected: No such path! Should throw a value error."
#     print "Did brute force search raise an error?", bruteRaisedErr
#     print "Did DFS search raise an error?", dfsRaisedErr

#     # Test case 8
#     print "---------------"
#     print "Test case 8:"
#     print "Find the shortest-path from Building 10 to 32 without walking"
#     print "more than 100 meters in total"
#     bruteRaisedErr = 'No'
#     dfsRaisedErr = 'No'
#     try:
#         bruteForceSearch(mitMap, '10', '32', 100, LARGE_DIST)
#     except ValueError:
#         bruteRaisedErr = 'Yes'

#     try:
#         directedDFS(mitMap, '10', '32', 100, LARGE_DIST)
#     except ValueError:
#         dfsRaisedErr = 'Yes'

#     print "Expected: No such path! Should throw a value error."
#     print "Did brute force search raise an error?", bruteRaisedErr
#     print "Did DFS search raise an error?", dfsRaisedErr
